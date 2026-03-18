"""API clients for enriching notes with academic paper and author data."""

import os
import requests
from urllib.parse import quote, urlparse

import yaml


def load_config() -> dict:
    """Load config.yaml from _system/ relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "_system", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


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


class BraveSearchClient:
    """Client for querying the Brave Search API."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    # Domains filtered out to favour high-signal sources
    BLOCKED_DOMAINS = {
        "reddit.com", "www.reddit.com",
        "quora.com", "www.quora.com",
        "pinterest.com", "www.pinterest.com",
        "tiktok.com", "www.tiktok.com",
        "facebook.com", "www.facebook.com",
        "instagram.com", "www.instagram.com",
        "twitter.com", "x.com",
        "yahoo.com", "answers.yahoo.com",
    }

    def __init__(self, api_key=None):
        config = load_config()
        self.api_key = api_key or config.get("brave_search_key", "")
        if not self.api_key:
            raise ValueError(
                "Brave API key required: pass api_key or set brave_search_key "
                "in _system/config.yaml"
            )
        self.session = requests.Session()
        self.session.headers["X-Subscription-Token"] = self.api_key
        self.session.headers["Accept"] = "application/json"

    def _domain_from_url(self, url):
        """Extract the root domain from a URL."""
        hostname = urlparse(url).hostname or ""
        return hostname.lower()

    def search_articles(self, query, max_results=3):
        """Search the web for articles on a topic, filtering low-signal sites.

        Args:
            query: Search query string.
            max_results: Number of high-quality results to return (default 3).

        Returns:
            List of dicts with keys: title, url, description, domain.
        """
        # Request extra results so we still hit max_results after filtering
        fetch_count = max_results + len(self.BLOCKED_DOMAINS)
        resp = self.session.get(
            self.BASE_URL,
            params={"q": query, "count": fetch_count},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            url = item.get("url", "")
            domain = self._domain_from_url(url)

            if domain in self.BLOCKED_DOMAINS:
                continue

            results.append({
                "title": item.get("title"),
                "url": url,
                "description": item.get("description"),
                "domain": domain,
            })

            if len(results) >= max_results:
                break

        return results


class Enricher:
    """Orchestrates OpenAlex, Crossref, and Brave clients to enrich LLM output."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        self.scholarly_enabled = cfg.get("scholarly_search", True)
        self.web_enabled = cfg.get("web_search", True)

        # Initialise clients lazily — only if their feature is enabled
        self.openalex = OpenAlexClient() if self.scholarly_enabled else None
        self.crossref = CrossrefClient() if self.scholarly_enabled else None
        self.brave = None
        if self.web_enabled:
            try:
                self.brave = BraveSearchClient()
            except ValueError:
                print("  [Enricher] Brave API key not configured — web search disabled")

    # ── Public API ──

    def enrich(self, llm_results: dict) -> dict:
        """Enrich LLM output with scholarly refs and author/person metadata.

        Args:
            llm_results: dict from LLMProcessor.process() with keys
                description_mining, transcript_processing, note_generation.

        Returns:
            The same dict, mutated in place, with enrichment fields populated:
            - note_generation.video_note.has_scholarly_refs (bool)
            - note_generation.video_note.scholarly_refs_markdown (str)
            - Each person in note_generation.people gets:
              h_index, publications_count, evidence_links
        """
        note_gen = llm_results.get("note_generation", {})
        video_note = note_gen.get("video_note", {})
        desc_mining = llm_results.get("description_mining", {})
        transcript = llm_results.get("transcript_processing", {})

        # 1. Scholarly references for the video note
        if self.scholarly_enabled:
            print("  [Enrich 1/2] Searching scholarly references...")
            refs = self._gather_scholarly_refs(video_note, desc_mining, transcript)
            video_note["has_scholarly_refs"] = len(refs) > 0
            video_note["scholarly_refs_markdown"] = self._format_refs_markdown(refs)
        else:
            print("  [Enrich 1/2] Scholarly search disabled — skipping")

        # 2. Person enrichment
        people = note_gen.get("people", [])
        if people:
            print(f"  [Enrich 2/2] Enriching {len(people)} people...")
            for person in people:
                self._enrich_person(person)
        else:
            print("  [Enrich 2/2] No people to enrich")

        return llm_results

    # ── Scholarly References ──

    def _build_search_query(self, video_note: dict, transcript: dict) -> str:
        """Build a search query from the video's primary topics and title."""
        topics = video_note.get("topics", [])
        # Use up to 3 primary/secondary topics
        primary = []
        for t in transcript.get("topics", []):
            if t.get("relevance") in ("primary", "secondary"):
                primary.append(t["name"])
            if len(primary) >= 3:
                break
        # Fallback to video_note topics if transcript topics not structured
        if not primary and topics:
            primary = topics[:3]
        return " ".join(primary) if primary else video_note.get("cleaned_title", "")

    def _gather_scholarly_refs(
        self, video_note: dict, desc_mining: dict, transcript: dict
    ) -> list[dict]:
        """Search OpenAlex and Crossref for papers related to the video."""
        query = self._build_search_query(video_note, transcript)
        if not query:
            return []

        refs = []
        seen_titles = set()

        # OpenAlex
        if self.openalex:
            try:
                works = self.openalex.search_works(query, max_results=3)
                for w in works:
                    key = (w.get("title") or "").lower().strip()
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        refs.append({
                            "title": w["title"],
                            "authors": w["authors"],
                            "doi": w["doi"],
                            "year": w.get("year"),
                            "source": "OpenAlex",
                        })
            except Exception as e:
                print(f"    OpenAlex search failed: {e}")

        # Crossref
        if self.crossref:
            try:
                works = self.crossref.resolve_doi(query, max_results=3)
                for w in works:
                    key = (w.get("title") or "").lower().strip()
                    if key and key not in seen_titles:
                        seen_titles.add(key)
                        refs.append({
                            "title": w["title"],
                            "authors": w["authors"],
                            "doi": w.get("doi"),
                            "year": w.get("publication_date", "").split("-")[0] or None,
                            "source": "Crossref",
                        })
            except Exception as e:
                print(f"    Crossref search failed: {e}")

        # Also flag any papers from description mining book_references
        for ref in desc_mining.get("book_references", []):
            if ref.get("type") == "paper":
                key = (ref.get("title") or "").lower().strip()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    refs.append({
                        "title": ref["title"],
                        "authors": [ref.get("authors", "")],
                        "doi": None,
                        "year": None,
                        "source": "description",
                    })

        return refs

    @staticmethod
    def _format_refs_markdown(refs: list[dict]) -> str:
        """Format scholarly references as markdown for the video note."""
        if not refs:
            return "*No scholarly references found.*"

        lines = []
        for i, ref in enumerate(refs, 1):
            title = ref.get("title") or "Untitled"
            authors = ref.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            year = ref.get("year")
            year_str = f" ({year})" if year else ""
            doi = ref.get("doi")
            if doi:
                doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
                lines.append(f"{i}. **{title}**{year_str} — {author_str} — [DOI]({doi_url})")
            else:
                lines.append(f"{i}. **{title}**{year_str} — {author_str}")
        return "\n".join(lines)

    # ── Person Enrichment ──

    def _enrich_person(self, person: dict) -> None:
        """Enrich a single person dict with h-index, publications, and evidence links."""
        name = person.get("name", "")
        if not name:
            return

        # OpenAlex author lookup
        if self.openalex and self.scholarly_enabled:
            try:
                author_data = self.openalex.get_author(name)
                if author_data:
                    person["h_index"] = author_data.get("h_index", "unknown")
                    person["publications_count"] = author_data.get("publication_count", "unknown")
                    print(f"    {name}: h-index={person['h_index']}, "
                          f"pubs={person['publications_count']}")
                else:
                    person.setdefault("h_index", "unknown")
                    person.setdefault("publications_count", "unknown")
                    print(f"    {name}: not found in OpenAlex")
            except Exception as e:
                person.setdefault("h_index", "unknown")
                person.setdefault("publications_count", "unknown")
                print(f"    {name}: OpenAlex lookup failed: {e}")
        else:
            person.setdefault("h_index", "unknown")
            person.setdefault("publications_count", "unknown")

        # Brave web search for evidence links
        if self.brave and self.web_enabled:
            role = person.get("role", "")
            query = f"{name} {role}".strip() if role else name
            try:
                articles = self.brave.search_articles(query, max_results=3)
                person["evidence_links"] = [
                    f"[{a['title']}]({a['url']})" for a in articles
                ]
                print(f"    {name}: {len(articles)} evidence links found")
            except Exception as e:
                person.setdefault("evidence_links", [])
                print(f"    {name}: Brave search failed: {e}")
        else:
            person.setdefault("evidence_links", [])


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

    # --- Brave Search tests ---
    try:
        brave = BraveSearchClient()
        print("\n\n=== Brave Search: 'large language model scaling laws' ===")
        articles = brave.search_articles("large language model scaling laws",
                                         max_results=3)
        for i, a in enumerate(articles, 1):
            print(f"\n--- Article {i} ---")
            print(f"  Title:       {a['title']}")
            print(f"  URL:         {a['url']}")
            print(f"  Domain:      {a['domain']}")
            desc_preview = (a['description'] or '')[:120]
            print(f"  Description: {desc_preview}...")
    except ValueError as e:
        print(f"\n\n=== Brave Search: SKIPPED ({e}) ===")
