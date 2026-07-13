from natsuki.query_understanding import expand_query


def test_appends_expansion_terms():
    def fake_chat(messages):
        return "cerebral\nneuroimaging\nmri"

    result = expand_query("brain scan", chat_fn=fake_chat)
    assert result == "brain scan cerebral neuroimaging mri"


def test_strips_blank_lines_and_whitespace():
    def fake_chat(messages):
        return "  cerebral  \n\n  neuroimaging\n   \n"

    result = expand_query("brain scan", chat_fn=fake_chat)
    assert result == "brain scan cerebral neuroimaging"


def test_falls_back_to_original_query_on_empty_response():
    def fake_chat(messages):
        return ""

    assert expand_query("brain scan", chat_fn=fake_chat) == "brain scan"


def test_falls_back_to_original_query_on_exception():
    def fake_chat(messages):
        raise RuntimeError("network error")

    assert expand_query("brain scan", chat_fn=fake_chat) == "brain scan"


def test_passes_query_as_user_message():
    captured = {}

    def fake_chat(messages):
        captured["messages"] = messages
        return "term1\nterm2"

    expand_query("my query", chat_fn=fake_chat)
    assert captured["messages"][-1] == {"role": "user", "content": "my query"}
