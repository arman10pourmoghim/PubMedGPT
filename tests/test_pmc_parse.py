from app.ncbi import NCBIClient

SAMPLE_PMC = """
<article>
  <front><article-meta>
    <article-id pub-id-type="pmcid">PMC9999999</article-id>
  </article-meta></front>
  <body>
    <sec><title>Methods</title><p>We enrolled 42 zebrafish.</p></sec>
    <sec><title>Results</title><p>Significant swim speed increase.</p></sec>
  </body>
</article>
"""

def test_parse_pmc_sections_basic():
    m = NCBIClient.parse_pmc_sections(SAMPLE_PMC)
    assert "9999999" in m
    sec = m["9999999"]
    assert "Methods" in sec and "Results" in sec
    assert "zebrafish" in sec["Methods"]
    assert "swim" in sec["Results"]
