import logging
import os
import zipfile
from datetime import datetime
from io import BytesIO
from typing import BinaryIO, Optional
from uuid import UUID

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from core.base import FileConfig, FileProvider, R2RException

logger = logging.getLogger()


class AzureBlobFileProvider(FileProvider):
    """Azure Blob Storage implementation of the FileProvider."""

    def __init__(self, config: FileConfig):
        super().__init__(config)

        self.container_name = self.config.azure_container_name or os.getenv(
            "AZURE_CONTAINER_NAME"
        )

        # Get connection credentials
        connection_string = (
            self.config.azure_storage_connection_string
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        account_name = self.config.azure_storage_account_name or os.getenv(
            "AZURE_STORAGE_ACCOUNT_NAME"
        )
        account_key = self.config.azure_storage_account_key or os.getenv(
            "AZURE_STORAGE_ACCOUNT_KEY"
        )

        # Initialize BlobServiceClient
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        elif account_name and account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=account_key,
            )
        else:
            raise R2RException(
                status_code=500,
                message="Azure Blob Storage requires either a connection string "
                "or account name and key",
            )

        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    def _get_blob_name(self, document_id: UUID) -> str:
        """Generate a unique blob name for a document."""
        return f"documents/{document_id}"

    async def initialize(self) -> None:
        """Initialize Azure Blob container."""
        try:
            if self.container_client.exists():
                logger.info(
                    f"Using existing Azure container: {self.container_name}"
                )
            else:
                logger.info(f"Creating Azure container: {self.container_name}")
                self.container_client.create_container()
        except ResourceExistsError:
            logger.info(
                f"Azure container already exists: {self.container_name}"
            )
        except Exception as e:
            logger.error(f"Error accessing Azure container: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to initialize Azure container: {e}",
            ) from e

    async def store_file(
        self,
        document_id: UUID,
        file_name: str,
        file_content: BytesIO,
        file_type: Optional[str] = None,
    ) -> None:
        """Store a file in Azure Blob Storage."""
        try:
            blob_name = self._get_blob_name(document_id)
            blob_client = self.container_client.get_blob_client(blob_name)

            file_content.seek(0)
            blob_client.upload_blob(
                file_content,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=file_type or "application/octet-stream"
                ),
                metadata={
                    "filename": file_name,
                    "document_id": str(document_id),
                },
            )

        except Exception as e:
            logger.error(f"Error storing file in Azure Blob Storage: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to store file in Azure Blob Storage: {e}",
            ) from e

    async def retrieve_file(
        self, document_id: UUID
    ) -> Optional[tuple[str, BinaryIO, int]]:
        """Retrieve a file from Azure Blob Storage."""
        blob_name = self._get_blob_name(document_id)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            blob_properties = blob_client.get_blob_properties()

            file_name = blob_properties.metadata.get(
                "filename", f"file-{document_id}"
            )
            file_size = blob_properties.size

            file_content = BytesIO()
            download_stream = blob_client.download_blob()
            file_content.write(download_stream.readall())

            file_content.seek(0)
            return file_name, file_content, file_size

        except ResourceNotFoundError as e:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            ) from e
        except Exception as e:
            raise R2RException(
                status_code=500,
                message=f"Error retrieving file from Azure Blob Storage: {e}",
            ) from e

    async def retrieve_files_as_zip(
        self,
        document_ids: Optional[list[UUID]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[str, BinaryIO, int]:
        """Retrieve multiple files from Azure and return them as a zip file."""
        if not document_ids:
            raise R2RException(
                status_code=400,
                message="Document IDs must be provided for Azure file retrieval",
            )

        zip_buffer = BytesIO()

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            for doc_id in document_ids:
                try:
                    result = await self.retrieve_file(doc_id)
                    if result:
                        file_name, file_content, _ = result

                        if hasattr(file_content, "getvalue"):
                            content_bytes = file_content.getvalue()
                        else:
                            file_content.seek(0)
                            content_bytes = file_content.read()

                        zip_file.writestr(file_name, content_bytes)

                except R2RException as e:
                    if e.status_code == 404:
                        logger.warning(
                            f"File for document {doc_id} not found, skipping"
                        )
                        continue
                    else:
                        raise

        zip_buffer.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"files_export_{timestamp}.zip"
        zip_size = zip_buffer.getbuffer().nbytes

        if zip_size == 0:
            raise R2RException(
                status_code=404,
                message="No files found for the specified document IDs",
            )

        return zip_filename, zip_buffer, zip_size

    async def delete_file(self, document_id: UUID) -> bool:
        """Delete a file from Azure Blob Storage."""
        blob_name = self._get_blob_name(document_id)
        blob_client = self.container_client.get_blob_client(blob_name)

        try:
            blob_client.get_blob_properties()
            blob_client.delete_blob()
            return True

        except ResourceNotFoundError as e:
            raise R2RException(
                status_code=404,
                message=f"File for document {document_id} not found",
            ) from e
        except Exception as e:
            logger.error(f"Error deleting file from Azure Blob Storage: {e}")
            raise R2RException(
                status_code=500,
                message=f"Failed to delete file from Azure Blob Storage: {e}",
            ) from e

    async def get_files_overview(
        self,
        offset: int,
        limit: int,
        filter_document_ids: Optional[list[UUID]] = None,
        filter_file_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get an overview of stored files.

        Note: Since Azure Blob Storage doesn't have native query capabilities
        like a database, this implementation works best when document IDs are
        provided.
        """
        results = []

        if filter_document_ids:
            for doc_id in filter_document_ids:
                blob_name = self._get_blob_name(doc_id)
                blob_client = self.container_client.get_blob_client(blob_name)
                try:
                    blob_properties = blob_client.get_blob_properties()

                    file_info = {
                        "document_id": doc_id,
                        "file_name": blob_properties.metadata.get(
                            "filename", f"file-{doc_id}"
                        ),
                        "file_key": blob_name,
                        "file_size": blob_properties.size,
                        "file_type": blob_properties.content_settings.content_type,
                        "created_at": blob_properties.creation_time,
                        "updated_at": blob_properties.last_modified,
                    }

                    results.append(file_info)
                except ResourceNotFoundError:
                    continue
        else:
            try:
                blobs = list(
                    self.container_client.list_blobs(
                        name_starts_with="documents/"
                    )
                )

                page_blobs = blobs[offset : offset + limit]

                for blob in page_blobs:
                    blob_name = blob.name
                    doc_id_str = blob_name.split("/")[-1]

                    try:
                        doc_id = UUID(doc_id_str)

                        blob_client = self.container_client.get_blob_client(
                            blob_name
                        )
                        blob_properties = blob_client.get_blob_properties()

                        file_name = blob_properties.metadata.get(
                            "filename", f"file-{doc_id}"
                        )

                        if (
                            filter_file_names
                            and file_name not in filter_file_names
                        ):
                            continue

                        file_info = {
                            "document_id": doc_id,
                            "file_name": file_name,
                            "file_key": blob_name,
                            "file_size": blob.size,
                            "file_type": blob_properties.content_settings.content_type,
                            "created_at": blob.creation_time,
                            "updated_at": blob.last_modified,
                        }

                        results.append(file_info)
                    except ValueError:
                        continue
            except Exception as e:
                logger.error(f"Error listing files in Azure container: {e}")
                raise R2RException(
                    status_code=500,
                    message=f"Failed to list files from Azure Blob Storage: {e}",
                ) from e

        if not results:
            raise R2RException(
                status_code=404,
                message="No files found with the given filters",
            )

        return results
