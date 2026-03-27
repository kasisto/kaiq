"""
Unit tests for _ensure_collections_exist in simple ingestion workflow.

Tests cover:
- Happy path: all collections exist and belong to user
- 403: collection exists but user doesn't own it
- 404: collection doesn't exist
- Deduplication of collection IDs
- Invalid response format from handler
- Empty collection list
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from core.base import R2RException
from core.main.orchestration.simple.ingestion_workflow import (
    simple_ingestion_factory,
)


def _make_document_info(owner_id=None, collection_ids=None):
    """Create a minimal DocumentResponse-like object."""
    doc = MagicMock()
    doc.id = uuid4()
    doc.owner_id = owner_id or uuid4()
    doc.collection_ids = collection_ids or []
    doc.title = "test-doc"
    return doc


def _make_collection(collection_id):
    """Create a minimal collection response object."""
    col = MagicMock()
    col.id = collection_id
    return col


def _make_user_info(collection_ids=None):
    """Create a minimal user info object."""
    user = MagicMock()
    user.collection_ids = collection_ids or []
    return user


def _build_factory():
    """Build a simple_ingestion_factory and extract _ensure_collections_exist."""
    service = MagicMock()
    service.providers = MagicMock()
    service.providers.database = MagicMock()
    service.providers.database.collections_handler = MagicMock()
    service.providers.database.users_handler = MagicMock()

    # Make async mocks
    service.providers.database.collections_handler.get_collections_overview = AsyncMock()
    service.providers.database.users_handler.get_user_by_id = AsyncMock()

    result = simple_ingestion_factory(service)

    # Extract _ensure_collections_exist from the closure of ingest_files.
    # The factory returns {"ingest-files": ingest_files, ...} and the inner
    # function captures _ensure_collections_exist as a closure variable.
    fn = result["ingest-files"]
    idx = fn.__code__.co_freevars.index("_ensure_collections_exist")
    ensure_fn = fn.__closure__[idx].cell_contents

    return service, ensure_fn


class TestEnsureCollectionsExist:
    """Tests for _ensure_collections_exist helper."""

    @pytest.fixture
    def setup(self):
        service, ensure_fn = _build_factory()
        return service, ensure_fn

    @pytest.mark.asyncio
    async def test_all_collections_owned_by_user(self, setup):
        """Collections exist and belong to user — should pass without error."""
        service, ensure_fn = setup
        col_id_1 = uuid4()
        col_id_2 = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": [_make_collection(col_id_1), _make_collection(col_id_2)]
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[col_id_1, col_id_2])
        )

        # Should not raise
        await ensure_fn(doc, [col_id_1, col_id_2])

        service.providers.database.collections_handler.get_collections_overview.assert_awaited_once()
        service.providers.database.users_handler.get_user_by_id.assert_awaited_once_with(
            id=owner_id
        )

    @pytest.mark.asyncio
    async def test_collection_not_owned_by_user_raises_403(self, setup):
        """Collection exists but user doesn't own it — should raise 403."""
        service, ensure_fn = setup
        col_id = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": [_make_collection(col_id)]
        }
        # User does NOT have this collection
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[])
        )

        with pytest.raises(R2RException) as exc_info:
            await ensure_fn(doc, [col_id])

        assert exc_info.value.status_code == 403
        assert "does not belong to the document owner" in str(exc_info.value.message)
        # Verify owner_id is NOT in the error message (M5 fix)
        assert str(owner_id) not in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_nonexistent_collection_raises_404(self, setup):
        """Collection doesn't exist — should raise 404, not auto-create."""
        service, ensure_fn = setup
        col_id = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        # No collections found
        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": []
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[])
        )

        with pytest.raises(R2RException) as exc_info:
            await ensure_fn(doc, [col_id])

        assert exc_info.value.status_code == 404
        assert "does not exist" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_deduplicates_collection_ids(self, setup):
        """Duplicate collection IDs should be deduplicated."""
        service, ensure_fn = setup
        col_id = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": [_make_collection(col_id)]
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[col_id])
        )

        # Pass duplicate IDs
        await ensure_fn(doc, [col_id, col_id, col_id])

        # Should query with deduplicated list
        call_kwargs = service.providers.database.collections_handler.get_collections_overview.call_args
        filter_ids = call_kwargs.kwargs.get("filter_collection_ids") or call_kwargs[1].get("filter_collection_ids")
        assert len(filter_ids) == 1

    @pytest.mark.asyncio
    async def test_empty_collection_ids(self, setup):
        """Empty collection list should be a no-op."""
        service, ensure_fn = setup
        doc = _make_document_info()

        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": []
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info()
        )

        # Should not raise
        await ensure_fn(doc, [])

    @pytest.mark.asyncio
    async def test_invalid_response_format_raises_500(self, setup):
        """Invalid response from collections handler should raise 500."""
        service, ensure_fn = setup
        doc = _make_document_info()

        # Return non-list results
        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": "not-a-list"
        }

        with pytest.raises(R2RException) as exc_info:
            await ensure_fn(doc, [uuid4()])

        assert exc_info.value.status_code == 500
        assert "Invalid response format" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_mixed_owned_and_unowned_raises_403(self, setup):
        """If user owns some but not all collections, should raise 403 on first unowned."""
        service, ensure_fn = setup
        owned_col = uuid4()
        unowned_col = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": [_make_collection(owned_col), _make_collection(unowned_col)]
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[owned_col])  # Only owns one
        )

        with pytest.raises(R2RException) as exc_info:
            await ensure_fn(doc, [owned_col, unowned_col])

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_mixed_existing_and_nonexistent_raises_404(self, setup):
        """If some collections exist and some don't, should raise 404 on the missing one."""
        service, ensure_fn = setup
        existing_col = uuid4()
        missing_col = uuid4()
        owner_id = uuid4()

        doc = _make_document_info(owner_id=owner_id)

        # Only one collection exists
        service.providers.database.collections_handler.get_collections_overview.return_value = {
            "results": [_make_collection(existing_col)]
        }
        service.providers.database.users_handler.get_user_by_id.return_value = (
            _make_user_info(collection_ids=[existing_col])
        )

        with pytest.raises(R2RException) as exc_info:
            await ensure_fn(doc, [existing_col, missing_col])

        assert exc_info.value.status_code == 404
