"""Minimal from-scratch tokenizer: lowercase, strip punctuation, drop stopwords."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, fixed stopword list (English). Not exhaustive by design -- the goal
# is to cut obvious noise terms, not to replicate a linguistics resource.
STOPWORDS = frozenset(
    """
    a an the and or but if while is are was were be been being
    to of in on at for with by from as into over under
    this that these those it its it's he she they we you i
    not no do does did doing have has had having
    will would shall should can could may might must
    there here what which who whom when where why how
    """.split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
