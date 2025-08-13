# tests/test_prompt_shape.py
from app.synthesis import build_messages

def test_build_messages_shape():
    msgs = build_messages(
        question="What is the link between sleep and glymphatic clearance?",
        contexts=[
            {"pmid": "123", "title": "Sleep and clearance", "text": "During sleep, clearance increases..."},
            {"pmid": "456", "title": "Glymphatic dynamics", "text": "The glymphatic system facilitates..."},
        ],
    )
    assert isinstance(msgs, list) and len(msgs) >= 2
    roles = [m["role"] for m in msgs]
    assert "system" in roles and "user" in roles
