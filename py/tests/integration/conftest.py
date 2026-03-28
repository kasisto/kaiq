import os
import uuid
import asyncio
import time
from typing import AsyncGenerator

import pytest

from r2r import R2RAsyncClient, R2RClient, R2RException


class RetryableR2RAsyncClient(R2RAsyncClient):
    """R2RAsyncClient with automatic retry logic for timeouts"""

    async def _make_request(self, method, endpoint, version="v3", **kwargs):
        retries = 0
        max_retries = 3
        delay = 1.0

        while True:
            try:
                return await super()._make_request(method, endpoint, version, **kwargs)
            except R2RException as e:
                if "Request failed" in str(e) and retries < max_retries:
                    retries += 1
                    wait_time = delay * (2 ** (retries - 1))
                    print(f"Request timed out. Retrying ({retries}/{max_retries}) after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                elif "429" in str(e) and retries < max_retries:
                    retries += 1
                    wait_time = delay * (3 ** (retries - 1))
                    print(f"Rate limited. Retrying ({retries}/{max_retries}) after {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

class RetryableR2RClient(R2RClient):
    """R2RClient with automatic retry logic for timeouts"""

    def _make_request(self, method, endpoint, version="v3", **kwargs):
        retries = 0
        max_retries = 3
        delay = 1.0

        while True:
            try:
                return super()._make_request(method, endpoint, version, **kwargs)
            except R2RException as e:
                if ("Request failed" in str(e) or "timed out" in str(e)) and retries < max_retries:
                    retries += 1
                    wait_time = delay * (2 ** (retries - 1))
                    print(f"Request timed out. Retrying ({retries}/{max_retries}) after {wait_time:.2f}s...")
                    time.sleep(wait_time)
                elif "429" in str(e) and retries < max_retries:
                    retries += 1
                    wait_time = delay * (3 ** (retries - 1))
                    print(f"Rate limited. Retrying ({retries}/{max_retries}) after {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    raise



class TestConfig:
    def __init__(self):
        self.base_url = "http://localhost:7272"
        self.index_wait_time = 1.0
        self.chunk_creation_wait_time = 1.0
        self.superuser_email = "admin@example.com"
        self.superuser_password = "change_me_immediately"
        self.test_timeout = 30  # seconds


# Change this to session scope to match the client fixture
@pytest.fixture(scope="session")
def config() -> TestConfig:
    return TestConfig()


@pytest.fixture(scope="session")
async def client(config) -> AsyncGenerator[R2RClient, None]:
    """Create a shared client instance for the test session."""
    yield RetryableR2RClient(config.base_url)


@pytest.fixture
def mutable_client(config) -> R2RClient:
    """Create a shared client instance for the test session."""
    return RetryableR2RClient(config.base_url)


@pytest.fixture
async def aclient(config) -> AsyncGenerator[R2RAsyncClient, None]:
    """Create a retryable client instance for the test session."""
    yield RetryableR2RAsyncClient(config.base_url)


@pytest.fixture
async def superuser_client(
        mutable_client: R2RClient,
        config: TestConfig) -> AsyncGenerator[R2RClient, None]:
    """Creates a superuser client for tests requiring elevated privileges."""
    await mutable_client.users.login(config.superuser_email, config.superuser_password)
    yield mutable_client
    await mutable_client.users.logout()


@pytest.fixture(scope="session")
def test_document(client: R2RClient):
    """Create and yield a test document, then clean up."""
    random_suffix = str(uuid.uuid4())
    doc_id = client.documents.create(
        raw_text=f"{random_suffix} Test doc for collections",
        run_with_orchestration=False,
    ).results.document_id

    yield doc_id
    # Cleanup: Try deleting the document if it still exists
    try:
        client.documents.delete(id=doc_id)
    except R2RException:
        pass


@pytest.fixture(scope="session")
def test_collection(client: R2RClient, test_document):
    """Create a test collection with sample documents and clean up after
    tests."""
    collection_name = f"Test Collection {uuid.uuid4()}"
    collection_id = client.collections.create(name=collection_name).results.id

    docs = [
        {
            "text":
            f"Aristotle was a Greek philosopher who studied under Plato {str(uuid.uuid4())}.",
            "metadata": {
                "rating": 5,
                "tags": ["philosophy", "greek"],
                "category": "ancient",
            },
        },
        {
            "text":
            f"Socrates is considered a founder of Western philosophy  {str(uuid.uuid4())}.",
            "metadata": {
                "rating": 3,
                "tags": ["philosophy", "classical"],
                "category": "ancient",
            },
        },
        {
            "text":
            f"Rene Descartes was a French philosopher. unique_philosopher  {str(uuid.uuid4())}",
            "metadata": {
                "rating": 8,
                "tags": ["rationalism", "french"],
                "category": "modern",
            },
        },
        {
            "text":
            f"Immanuel Kant, a German philosopher, influenced Enlightenment thought  {str(uuid.uuid4())}.",
            "metadata": {
                "rating": 7,
                "tags": ["enlightenment", "german"],
                "category": "modern",
            },
        },
    ]

    doc_ids = []
    for doc in docs:
        doc_id = client.documents.create(
            raw_text=doc["text"], metadata=doc["metadata"]).results.document_id
        doc_ids.append(doc_id)
        client.collections.add_document(collection_id, doc_id)
    client.collections.add_document(collection_id, test_document)

    yield {"collection_id": collection_id, "document_ids": doc_ids}

    # Cleanup after tests
    try:
        # Remove and delete all documents
        for doc_id in doc_ids:
            try:
                client.documents.delete(id=doc_id)
            except R2RException:
                pass
        # Delete the collection
        try:
            client.collections.delete(collection_id)
        except R2RException:
            pass
    except Exception as e:
        print(f"Error during test_collection cleanup: {e}")


# ---------------------------------------------------------------------------
# Database-layer fixtures (for test_db_*.py — direct DB handler tests)
# ---------------------------------------------------------------------------

from core.base import AppConfig, DatabaseConfig, VectorQuantizationType
from core.providers import NaClCryptoConfig, NaClCryptoProvider
from core.providers.database.postgres import (
    PostgresChunksHandler,
    PostgresCollectionsHandler,
    PostgresConversationsHandler,
    PostgresDatabaseProvider,
    PostgresDocumentsHandler,
    PostgresGraphsHandler,
    PostgresLimitsHandler,
    PostgresPromptsHandler,
)
from core.providers.database.users import PostgresUserHandler

TEST_DB_CONNECTION_STRING = os.environ.get(
    "TEST_DB_CONNECTION_STRING",
    "postgresql://postgres:postgres@localhost:5432/test_db",
)


@pytest.fixture
async def db_provider():
    crypto_provider = NaClCryptoProvider(NaClCryptoConfig(app={}))
    db_config = DatabaseConfig(
        app=AppConfig(project_name="test_project"),
        provider="postgres",
        connection_string=TEST_DB_CONNECTION_STRING,
        postgres_configuration_settings={
            "max_connections": 10,
            "statement_cache_size": 100,
        },
        project_name="test_project",
    )

    dimension = 4
    quantization_type = VectorQuantizationType.FP32

    provider = PostgresDatabaseProvider(
        db_config, dimension, crypto_provider, quantization_type
    )

    await provider.initialize()
    yield provider
    await provider.close()


@pytest.fixture
def crypto_provider():
    return NaClCryptoProvider(NaClCryptoConfig(app={}))


@pytest.fixture
async def chunks_handler(db_provider):
    handler = PostgresChunksHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        dimension=db_provider.dimension,
        quantization_type=db_provider.quantization_type,
    )
    await handler.create_tables()
    return handler


@pytest.fixture
async def collections_handler(db_provider):
    handler = PostgresCollectionsHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        config=db_provider.config,
    )
    await handler.create_tables()
    return handler


@pytest.fixture
async def conversations_handler(db_provider):
    handler = PostgresConversationsHandler(
        db_provider.project_name, db_provider.connection_manager
    )
    await handler.create_tables()
    return handler


@pytest.fixture
async def documents_handler(db_provider):
    handler = PostgresDocumentsHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        dimension=db_provider.dimension,
    )
    await handler.create_tables()
    return handler


@pytest.fixture
async def graphs_handler(db_provider):
    create_col_sql = f"""
        ALTER TABLE "{db_provider.project_name}"."graphs_entities"
        ADD COLUMN IF NOT EXISTS collection_ids UUID[] DEFAULT '{{}}';
    """
    await db_provider.connection_manager.execute_query(create_col_sql)

    handler = PostgresGraphsHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        dimension=db_provider.dimension,
        quantization_type=db_provider.quantization_type,
        collections_handler=None,
    )
    await handler.create_tables()
    return handler


@pytest.fixture
async def limits_handler(db_provider):
    handler = PostgresLimitsHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        config=db_provider.config,
    )
    await handler.create_tables()
    await db_provider.connection_manager.execute_query(
        f"TRUNCATE {handler._get_table_name('request_log')};"
    )
    return handler


@pytest.fixture
async def users_handler(db_provider, crypto_provider):
    handler = PostgresUserHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        crypto_provider=crypto_provider,
    )
    await handler.create_tables()
    await db_provider.connection_manager.execute_query(
        f"TRUNCATE {handler._get_table_name('users')} CASCADE;"
    )
    await db_provider.connection_manager.execute_query(
        f"TRUNCATE {handler._get_table_name('users_api_keys')} CASCADE;"
    )
    return handler


@pytest.fixture
async def prompt_handler(db_provider):
    handler = PostgresPromptsHandler(
        project_name=db_provider.project_name,
        connection_manager=db_provider.connection_manager,
        prompt_directory=None,
    )
    await handler.create_tables()
    return handler
