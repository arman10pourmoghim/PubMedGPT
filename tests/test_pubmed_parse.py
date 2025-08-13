# tests/test_pubmed_parse.py
from app.ncbi import NCBIClient

SAMPLE_EFETCH_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Abstract>
          <AbstractText Label="Background">Cats are mysterious.</AbstractText>
          <AbstractText Label="Methods">We observed 10 cats.</AbstractText>
          <AbstractText Label="Results">They ignored us.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
""".strip()

def test_parse_pubmed_abstracts_merges_segments():
    parsed = NCBIClient.parse_pubmed_abstracts(SAMPLE_EFETCH_XML)
    assert "12345678" in parsed
    assert "Cats are mysterious." in parsed["12345678"]
    assert "We observed 10 cats." in parsed["12345678"]
    assert "They ignored us." in parsed["12345678"]
