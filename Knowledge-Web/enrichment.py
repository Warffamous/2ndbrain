"""OpenAlex API client for enriching notes with academic paper and author data."""

import requests
from urllib.parse import quote


class OpenAlexClient:
    """Client for querying the OpenAlex academic database API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email=None):
        self.session = requests.Session()
        # OpenAlex asks for a polite pool email for better rate limits
        if email:
            self.session.params = {"mailto": email}

    def search_works(self, keywords, max_results=5):
        """Search for academic papers by keywords.

        Args:
            keywords: Search query string.
            max_results: Maximum number of results to return (default 5).

        Returns:
            List of dicts with keys: title, authors, doi, year, abstract.
        """
        resp = self.session.get(
            f"{self.BASE_URL}/works",
            params={"search": keywords, "per_page": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for work in data.get("results", []):
            # Reconstruct abstract from inverted index if available
            abstract = None
            inverted = work.get("abstract_inverted_index")
            if inverted:
                word_positions = []
                for word, positions in inverted.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)

            authors = [
                a.get("author", {}).get("display_name", "")
                for a in work.get("authorships", [])
            ]

            results.append({
                "title": work.get("title"),
                "authors": authors,
                "doi": work.get("doi"),
                "year": work.get("publication_year"),
                "abstract": abstract,
            })

        return results

    def get_author(self, name):
        """Look up an author by name.

        Args:
            name: Author name to search for.

        Returns:
            Dict with keys: h_index, publication_count, institution,
            google_scholar_url.
        """
        resp = self.session.get(
            f"{self.BASE_URL}/authors",
            params={"search": name, "per_page": 1},
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        author = results[0]
        summary = author.get("summary_stats", {})

        # Get most recent affiliated institution
        institution = None
        last_known = author.get("last_known_institutions", [])
        if last_known:
            institution = last_known[0].get("display_name")

        # Google Scholar URL from ids (not currently provided by OpenAlex,
        # but we check in case it becomes available)
        ids = author.get("ids", {})
        google_scholar_url = ids.get("google_scholar")

        return {
            "h_index": summary.get("h_index"),
            "publication_count": author.get("works_count"),
            "institution": institution,
            "google_scholar_url": google_scholar_url,
        }


if __name__ == "__main__":
    client = OpenAlexClient()

    print("=== Search Works: 'transformer attention mechanism' ===")
    papers = client.search_works("transformer attention mechanism", max_results=3)
    for i, p in enumerate(papers, 1):
        print(f"\n--- Paper {i} ---")
        print(f"  Title:   {p['title']}")
        print(f"  Authors: {', '.join(p['authors'][:3])}")
        print(f"  DOI:     {p['doi']}")
        print(f"  Year:    {p['year']}")
        abstract_preview = (p['abstract'] or '')[:120]
        print(f"  Abstract: {abstract_preview}...")

    print("\n\n=== Get Author: 'Yoshua Bengio' ===")
    author = client.get_author("Yoshua Bengio")
    if author:
        print(f"  H-index:        {author['h_index']}")
        print(f"  Publications:   {author['publication_count']}")
        print(f"  Institution:    {author['institution']}")
        print(f"  Google Scholar: {author['google_scholar_url']}")
    else:
        print("  No author found.")
