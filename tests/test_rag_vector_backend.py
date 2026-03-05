from app import main
from app.schemas import DatasetCreateRequest, RagQueryRequest


def setup_function():
    main.REQUEST_ID.set("rid-rag-backend")
    main.DATASETS.clear()
    main.RAG_INDEXES.clear()
    main.RAG_RUNTIME_FAISS_INDEX.clear()
    main._init_rag_vector_backend()


def test_rag_vector_backend_endpoint_shape():
    payload = main.v1_rag_vector_backend()
    assert payload["request_id"] == "rid-rag-backend"
    assert payload["configured_backend"] in {"in_memory", "faiss"}
    assert payload["active_backend"] in {"in_memory", "faiss"}
    assert isinstance(payload["vector_dim"], int)


def test_admin_rag_vector_backend_endpoint_shape():
    payload = main.v1_admin_rag_vector_backend()
    assert payload["request_id"] == "rid-rag-backend"
    assert "fallback_reason" in payload


def test_rag_query_returns_top_chunk_matching_query():
    created = main.v1_create_dataset(
        DatasetCreateRequest(
            name="rag-ranking-ds",
            type="rag_corpus",
            records=[
                {"text": "apple orchard maintenance guide"},
                {"text": "banana export policy and compliance checklist"},
                {"text": "pear storage details"},
            ],
        )
    )

    result = main.v1_rag_query(RagQueryRequest(dataset_id=created.dataset_id, query="banana policy", top_k=2))
    assert result.retrieved_chunks
    top = result.retrieved_chunks[0]
    assert "banana" in str(top["text"]).lower()


def test_faiss_backend_falls_back_when_dependencies_missing():
    prev_backend = main.VECTOR_INDEX_BACKEND
    prev_faiss = main.FAISS_AVAILABLE
    prev_numpy = main.NUMPY_AVAILABLE
    try:
        main.VECTOR_INDEX_BACKEND = "faiss"
        main.FAISS_AVAILABLE = False
        main.NUMPY_AVAILABLE = False
        backend, reason = main._resolve_rag_vector_backend()
        assert backend == "in_memory"
        assert reason is not None
    finally:
        main.VECTOR_INDEX_BACKEND = prev_backend
        main.FAISS_AVAILABLE = prev_faiss
        main.NUMPY_AVAILABLE = prev_numpy
        main._init_rag_vector_backend()
