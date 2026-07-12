from natsuki.bm25 import BM25
from natsuki.index.inverted_index import build_index

TOY_CORPUS = [
    ("d1", "the cat sat on the mat"),
    ("d2", "the dog sat on the log"),
    ("d3", "cats and dogs are great pets"),
    ("d4", "cat cat cat cat cat"),
]


def _index():
    return build_index(TOY_CORPUS, show_progress=False)


def test_only_documents_containing_a_query_term_are_returned():
    bm25 = BM25(_index())
    results = bm25.search("cat", top_k=10)
    doc_ids = {doc_id for doc_id, _ in results}
    # "cats" in d3 doesn't match the unstemmed token "cat"
    assert doc_ids == {"d1", "d4"}


def test_higher_term_frequency_ranks_higher():
    bm25 = BM25(_index())
    results = bm25.search("cat", top_k=10)
    ranked_ids = [doc_id for doc_id, _ in results]
    assert ranked_ids[0] == "d4"  # 5 occurrences of "cat" vs 1 in d1


def test_no_matching_terms_returns_empty():
    bm25 = BM25(_index())
    assert bm25.search("zzz_nonexistent_term", top_k=10) == []


def test_finalize_required_before_scoring():
    from natsuki.index.inverted_index import InvertedIndex

    idx = InvertedIndex()
    idx.add_document("d1", "hello world")
    try:
        BM25(idx)
        assert False, "expected RuntimeError for unfinalized index"
    except RuntimeError:
        pass
