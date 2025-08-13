from app.textproc import split_sentences, chunk_by_chars
from app.ranking import bm25_scores

def test_sentence_split_and_chunking():
    s = "Alpha beats beta. Gamma is neutral! Delta? Epsilon continues. Zeta ends."
    sents = split_sentences(s)
    assert len(sents) >= 4
    chunks = chunk_by_chars(s, max_chars=40, overlap=8)
    assert len(chunks) >= 2
    assert all("text" in c and "start_char" in c for c in chunks)

def test_bm25_basic_signal():
    docs = [
        "cats purr softly and sleep a lot",
        "dogs bark loudly and run quickly",
        "quantum cats may or may not purr"
    ]
    q = "cats purr"
    scores = bm25_scores(q, docs)
    # doc 0 and 2 should outrank the dog doc
    assert scores[0] > scores[1] or scores[2] > scores[1]
