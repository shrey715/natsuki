from natsuki.tokenizer import tokenize


def test_lowercases_and_splits_on_punctuation():
    assert tokenize("Cats, Dogs & Birds!", remove_stopwords=False) == [
        "cats",
        "dogs",
        "birds",
    ]


def test_removes_stopwords_by_default():
    assert tokenize("the cat sat on the mat") == ["cat", "sat", "mat"]


def test_keeps_stopwords_when_disabled():
    tokens = tokenize("the cat sat on the mat", remove_stopwords=False)
    assert "the" in tokens
    assert "on" in tokens
