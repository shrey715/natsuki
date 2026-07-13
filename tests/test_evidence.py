import numpy as np

from natsuki.evidence import extract_evidence, rank_sentences, split_sentences


def test_split_sentences_basic():
    text = "Cats are mammals. Dogs are loyal. Fish live in water."
    assert split_sentences(text) == [
        "Cats are mammals.",
        "Dogs are loyal.",
        "Fish live in water.",
    ]


def test_split_sentences_handles_question_and_exclamation():
    text = "Is this true? Yes it is! Great."
    assert split_sentences(text) == ["Is this true?", "Yes it is!", "Great."]


def test_split_sentences_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_rank_sentences_orders_by_similarity():
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    sentence_vectors = np.array(
        [
            [0.0, 1.0],  # orthogonal -- irrelevant
            [1.0, 0.0],  # identical -- most relevant
            [0.70710678, 0.70710678],  # halfway
        ],
        dtype=np.float32,
    )
    ranked = rank_sentences(query_vector, sentence_vectors)
    assert [idx for idx, _ in ranked] == [1, 2, 0]


def test_extract_evidence_picks_the_matching_sentence():
    doc_text = "Cats are mammals. The stock market rallied today. Water boils at 100 degrees."

    # fake embeddings: hand-picked so sentence 1 (stock market) exactly
    # matches the query vector, others don't.
    def fake_embed_query(query: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def fake_embed_documents(sentences: list[str]) -> np.ndarray:
        vectors = {
            "Cats are mammals.": [1.0, 0.0, 0.0],
            "The stock market rallied today.": [0.0, 1.0, 0.0],
            "Water boils at 100 degrees.": [0.0, 0.0, 1.0],
        }
        return np.array([vectors[s] for s in sentences], dtype=np.float32)

    results = extract_evidence(
        "market news",
        doc_text,
        top_n=1,
        embed_query_fn=fake_embed_query,
        embed_documents_fn=fake_embed_documents,
    )
    assert results == [("The stock market rallied today.", 1.0)]


def test_extract_evidence_top_n():
    doc_text = "A. B. C."

    def fake_embed_query(query: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def fake_embed_documents(sentences: list[str]) -> np.ndarray:
        # "A." most similar, then "B.", then "C."
        vectors = {"A.": [1.0, 0.0, 0.0], "B.": [0.9, 0.1, 0.0], "C.": [0.0, 1.0, 0.0]}
        return np.array([vectors[s] for s in sentences], dtype=np.float32)

    results = extract_evidence(
        "query",
        doc_text,
        top_n=2,
        embed_query_fn=fake_embed_query,
        embed_documents_fn=fake_embed_documents,
    )
    assert [sentence for sentence, _ in results] == ["A.", "B."]


def test_extract_evidence_empty_doc_returns_empty():
    results = extract_evidence(
        "query",
        "",
        embed_query_fn=lambda q: np.zeros(3, dtype=np.float32),
        embed_documents_fn=lambda ss: np.zeros((0, 3), dtype=np.float32),
    )
    assert results == []
