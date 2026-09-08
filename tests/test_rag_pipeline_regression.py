import asyncio
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import settings
from src.llm.model_factory import LLMFactory
from src.llm.prompt_templates import build_prompt
from src.llm.service import LLMService
from src.rag_flow.context_retriever import ContextRetriever
from src.rag_flow.retriever import Retriever
from src.rag_flow.reasoning_chain import ReasoningChain


class RecordingRetriever:
    def __init__(self, contexts):
        self.contexts = contexts
        self.queries = []

    def retrieve(self, query, top_k=None):
        self.queries.append((query, top_k or settings.TOP_K))
        return list(self.contexts)


class RecordingCompletions:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            for value in self.answer:
                yield type(
                    "Chunk",
                    (),
                    {"choices": [type("Choice", (), {"delta": type("Delta", (), {"content": value})()})()]},
                )()

        return stream()


def _collect(stream):
    async def collect():
        return [value async for value in stream]

    return asyncio.run(collect())


@pytest.mark.parametrize(
    ("provider", "model", "url"),
    [
        ("local", "local-test-model", "http://localhost:11434/v1"),
        ("gemini", "gemini-test-model", None),
    ],
)
def test_complete_rag_flow_retrieves_before_provider_generation(
    monkeypatch, provider, model, url
):
    query = "Which document describes the fire safety inspection procedure?"
    contexts = [
        {
            "text": "RETRIEVED-DOC-001: inspect the premises before issuing the record.",
            "metadata": {"document_id": "RETRIEVED-DOC-001"},
            "score": 0.95,
        },
        {
            "text": "RETRIEVED-DOC-002: retain the inspection result.",
            "metadata": {"document_id": "RETRIEVED-DOC-002"},
            "score": 0.91,
        },
    ]
    recording_retriever = RecordingRetriever(contexts)
    context_retriever = ContextRetriever.__new__(ContextRetriever)
    context_retriever.retriever = recording_retriever
    context_retriever.analyzer = type(
        "Analyzer", (), {"extract_entities": lambda self, value: {"query": value}}
    )()
    context_retriever.grader = type(
        "Grader", (), {"is_sufficient": lambda self, results, value: bool(results)}
    )()

    retrieved = context_retriever.get_context(query)
    assert recording_retriever.queries == [(query, settings.TOP_K)]
    assert retrieved["contexts"] == contexts

    context_text = "\n---\n".join(item["text"] for item in retrieved["contexts"])
    prompt = build_prompt(context_text, query)
    assert "RETRIEVED-DOC-001" in prompt
    assert "RETRIEVED-DOC-002" in prompt
    assert query in prompt

    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, "GEMINI_MODEL", model)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    if url:
        monkeypatch.setattr(settings, "LOCAL_LLM_URL", url)
        monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", model)

    selected_client = LLMFactory.get_llm()
    if provider == "local":
        assert str(selected_client.base_url).rstrip("/").endswith("localhost:11434/v1")
    else:
        assert "generativelanguage.googleapis.com" in str(selected_client.base_url)

    completions = RecordingCompletions(["answer", " grounded in retrieval"])
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    service = LLMService(client=client)
    answer = "".join(
        _collect(service.stream_chat([{"role": "user", "content": prompt}]))
    )

    assert answer == "answer grounded in retrieval"
    assert LLMFactory.get_model_name() == model
    request = completions.calls[0]
    assert request["model"] == model
    assert context_text in request["messages"][0]["content"]


def test_prompt_is_provider_independent(monkeypatch):
    context = "stable retrieved legal context"
    query = "What is the required procedure?"
    prompts = []
    for provider in ("local", "gemini"):
        monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
        if provider == "local":
            monkeypatch.setattr(settings, "LOCAL_LLM_URL", "http://localhost:11434/v1")
            monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "local-test-model")
        prompts.append(build_prompt(context, query))

    assert prompts[0] == prompts[1]


def test_provider_switch_does_not_replace_vector_index(monkeypatch):
    class FakeEmbeddingModel:
        def encode(self, values, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class FakeIndex:
        def __init__(self):
            self.calls = []

        def search(self, embeddings, top_k):
            self.calls.append((embeddings.copy(), top_k))
            return np.array([[0.99]], dtype=np.float32), np.array([[0]], dtype=np.int64)

    index = FakeIndex()
    retriever = Retriever.__new__(Retriever)
    retriever.model = FakeEmbeddingModel()
    retriever.index = index
    retriever.metadata = [{"text": "stable chunk", "metadata": {"document_id": "doc-1"}}]
    monkeypatch.setattr(settings, "SIMILARITY_THRESHOLD", 0.70)
    monkeypatch.setattr(settings, "TOP_K", 1)
    monkeypatch.setattr("src.rag_flow.retriever.torch.cuda.is_available", lambda: False)

    for provider in ("local", "gemini"):
        monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
        if provider == "local":
            monkeypatch.setattr(settings, "LOCAL_LLM_URL", "http://localhost:11434/v1")
            monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "local-test-model")
        assert retriever.retrieve("same query")
        assert retriever.index is index

    assert len(index.calls) == 2
    assert np.array_equal(index.calls[0][0], index.calls[1][0])


def test_reasoning_allows_rag_generation_without_optional_intent_metadata():
    result = ReasoningChain().process(
        "What is the procedure?",
        intent=None,
        entities={},
        chunks=[
            {
                "text": "Retrieved legal procedure",
                "metadata": {"category": "administrative"},
            }
        ],
    )

    assert result == {"ready": True}
