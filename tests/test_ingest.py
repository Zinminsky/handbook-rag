def test_all_documents_loaded(docs):
    assert len(docs) == 6
    assert {d.metadata["source"] for d in docs} >= {"01_leave_policy.md", "03_security_policy.md"}


def test_chunks_respect_size_and_keep_metadata(chunks):
    assert len(chunks) > 6
    for c in chunks:
        assert len(c.page_content) <= 500
        assert c.metadata["source"].endswith(".md")
        assert c.metadata["section"]            # section title from the markdown headers
        assert ":" in c.metadata["chunk_id"]


def test_chunk_ids_unique(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_sections_are_not_mixed_across_documents(chunks):
    sick = [c for c in chunks if "doctor's note" in c.page_content]
    assert sick and all(c.metadata["source"] == "01_leave_policy.md" for c in sick)
