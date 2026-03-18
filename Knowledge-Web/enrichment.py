"""API clients for enriching notes with academic paper and author data."""

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


class CrossrefClient:
    """Client for querying the Crossref metadata API."""

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email=None):
        self.session = requests.Session()
        # Crossref polite pool: pass email in User-Agent for better rate limits
        ua = "2ndBrain/0.1 (https://github.com/2ndbrain)"
        if email:
            ua += f" (mailto:{email})"
        self.session.headers["User-Agent"] = ua

    def resolve_doi(self, query, max_results=3):
        """Search Crossref for works matching a query string.

        Args:
            query: Search query (topic keyword, title fragment, etc.).
            max_results: Maximum number of results to return (default 3).

        Returns:
            List of dicts with keys: title, authors, doi, publisher,
            publication_date.
        """
        resp = self.session.get(
            f"{self.BASE_URL}/works",
            params={"query": query, "rows": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("message", {}).get("items", []):
            # Title is an array; take the first entry
            titles = item.get("title", [])
            title = titles[0] if titles else None

            # Authors: each has 'given' and 'family' keys
            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                authors.append(f"{given} {family}".strip())

            # Publication date from 'published' or 'issued' date-parts
            pub_date = None
            date_field = item.get("published") or item.get("issued")
            if date_field:
                parts = date_field.get("date-parts", [[]])
                if parts and parts[0]:
                    p = parts[0]
                    pub_date = "-".join(str(x) for x in p)

            results.append({
                "title": title,
                "authors": authors,
                "doi": item.get("DOI"),
                "publisher": item.get("publisher"),
                "publication_date": pub_date,
            })

        return results


if __name__ == "__main__":
    # --- OpenAlex tests ---
    client = OpenAlexClient()

    print("=== OpenAlex: Search Works 'transformer attention mechanism' ===")
    papers = client.search_works("transformer attention mechanism", max_results=3)
    for i, p in enumerate(papers, 1):
        print(f"\n--- Paper {i} ---")
        print(f"  Title:   {p['title']}")
        print(f"  Authors: {', '.join(p['authors'][:3])}")
        print(f"  DOI:     {p['doi']}")
        print(f"  Year:    {p['year']}")
        abstract_preview = (p['abstract'] or '')[:120]
        print(f"  Abstract: {abstract_preview}...")

    print("\n\n=== OpenAlex: Get Author 'Yoshua Bengio' ===")
    author = client.get_author("Yoshua Bengio")
    if author:
        print(f"  H-index:        {author['h_index']}")
        print(f"  Publications:   {author['publication_count']}")
        print(f"  Institution:    {author['institution']}")
        print(f"  Google Scholar: {author['google_scholar_url']}")
    else:
        print("  No author found.")

    # --- Crossref tests ---
    cr = CrossrefClient()

    print("\n\n=== Crossref: resolve_doi 'deep reinforcement learning' ===")
    works = cr.resolve_doi("deep reinforcement learning", max_results=3)
    for i, w in enumerate(works, 1):
        print(f"\n--- Result {i} ---")
        print(f"  Title:     {w['title']}")
        print(f"  Authors:   {', '.join(w['authors'][:3])}")
        print(f"  DOI:       {w['doi']}")
        print(f"  Publisher: {w['publisher']}")
        print(f"  Date:      {w['publication_date']}")
