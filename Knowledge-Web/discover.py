#!/usr/bin/env python3
"""
discover.py — Topic Discovery System

Scans Topic notes with discover_enabled: true and finds new related content
across three sources: OpenAlex, Brave Search, and YouTube Search API.

Deduplicates results against queue.db and existing vault notes, then
generates a Discovery Report markdown file at _system/discovery/YYYY-MM-DD.md.

Auto-ingest rules (from config.yaml discover settings):
  - YouTube videos above min view threshold → auto-queued with source='discover'
  - Articles and papers → flagged for manual approval in the report

Quota budget:
  - YouTube search.list = 100 units per topic searched
  - OpenAlex, Crossref, Brave = free / API-key based, no YouTube quota cost
  - With topics_per_run=20, worst case = 2000 YouTube units per daily run

Usage:
    python discover.py                    # Run discovery for all enabled topics
    python discover.py --topics 5         # Limit to 5 topics this run
    python discover.py --dry-run          # Generate report without auto-ingesting
    python discover.py --list-enabled     # Show which topics have discovery on
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import yaml

from queue_db import QueueDB
from enrichment import OpenAlexClient, CrossrefClient, BraveSearchClient, load_config
from vault_index import parse_frontmatter, load_existing_index


# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════

def get_vault_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_db(vault_path: str) -> QueueDB:
    db_path = os.path.join(vault_path, "_system", "queue.db")
    db = QueueDB(db_path)
    db.init_db()
    return db


# ═══════════════════════════════════════════
#  Topic Scanner
# ═══════════════════════════════════════════

def scan_discoverable_topics(vault_path: str) -> list[dict]:
    """
    Scan the Topics folder for notes with discover_enabled: true.

    Returns list of dicts with keys: concept, definition, related_topics, filepath.
    """
    topics_folder = os.path.join(vault_path, "Topics")
    if not os.path.exists(topics_folder):
        return []

    topics = []
    for filename in sorted(os.listdir(topics_folder)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(topics_folder, filename)
        fm = parse_frontmatter(filepath)
        if not fm:
            continue

        if fm.get("discover_enabled") is True:
            topics.append({
                "concept": fm.get("concept", filename.replace(".md", "")),
                "definition": fm.get("definition", ""),
                "related_topics": fm.get("related_topics", []),
                "filepath": filepath,
            })

    return topics


# ═══════════════════════════════════════════
#  Deduplication
# ═══════════════════════════════════════════

def build_known_set(db: QueueDB, vault_index: dict) -> set[str]:
    """
    Build a set of known identifiers for deduplication.

    Includes:
      - All video IDs currently in queue.db (any status)
      - All video note titles from vault-index.json
      - All article note titles from vault-index.json
    """
    known = set()

    # All video IDs from queue
    conn = db._connect()
    rows = conn.execute("SELECT id FROM queue").fetchall()
    for r in rows:
        known.add(r["id"])

    # Video titles from vault index
    for title in vault_index.get("notes", {}).get("videos", []):
        known.add(title.lower())

    # Article titles from vault index
    for title in vault_index.get("notes", {}).get("articles", []):
        known.add(title.lower())

    return known


def is_duplicate(item: dict, known: set[str]) -> bool:
    """Check if a discovery result is already known."""
    # Check video ID
    video_id = item.get("video_id")
    if video_id and video_id in known:
        return True

    # Check DOI
    doi = item.get("doi")
    if doi and doi.lower() in known:
        return True

    # Check title similarity (exact lowercase match)
    title = item.get("title", "")
    if title and title.lower().strip() in known:
        return True

    return False


# ═══════════════════════════════════════════
#  Source: OpenAlex (Papers)
# ═══════════════════════════════════════════

def search_openalex(
    client: OpenAlexClient,
    topic: str,
    max_results: int = 5,
) -> list[dict]:
    """Search OpenAlex for papers on a topic."""
    try:
        works = client.search_works(topic, max_results=max_results)
    except Exception as e:
        print(f"    OpenAlex error: {e}")
        return []

    results = []
    for w in works:
        doi = w.get("doi")
        results.append({
            "source": "openalex",
            "type": "paper",
            "title": w.get("title", "Untitled"),
            "authors": w.get("authors", []),
            "doi": doi,
            "year": w.get("year"),
            "url": doi if doi and doi.startswith("http") else
                   f"https://doi.org/{doi}" if doi else None,
            "abstract": (w.get("abstract") or "")[:200],
        })
    return results


# ═══════════════════════════════════════════
#  Source: Crossref (Papers — dedup with OpenAlex)
# ═══════════════════════════════════════════

def search_crossref(
    client: CrossrefClient,
    topic: str,
    max_results: int = 3,
) -> list[dict]:
    """Search Crossref for papers on a topic."""
    try:
        works = client.resolve_doi(topic, max_results=max_results)
    except Exception as e:
        print(f"    Crossref error: {e}")
        return []

    results = []
    for w in works:
        doi = w.get("doi")
        results.append({
            "source": "crossref",
            "type": "paper",
            "title": w.get("title", "Untitled"),
            "authors": w.get("authors", []),
            "doi": doi,
            "year": (w.get("publication_date") or "").split("-")[0] or None,
            "url": f"https://doi.org/{doi}" if doi else None,
        })
    return results


# ═══════════════════════════════════════════
#  Source: Brave Search (Articles)
# ═══════════════════════════════════════════

def search_brave(
    client: BraveSearchClient,
    topic: str,
    max_results: int = 3,
) -> list[dict]:
    """Search Brave for articles on a topic."""
    try:
        articles = client.search_articles(
            f"{topic} research OR study OR review",
            max_results=max_results,
        )
    except Exception as e:
        print(f"    Brave error: {e}")
        return []

    results = []
    for a in articles:
        results.append({
            "source": "brave",
            "type": "article",
            "title": a.get("title", "Untitled"),
            "url": a.get("url", ""),
            "domain": a.get("domain", ""),
            "description": a.get("description", ""),
        })
    return results


# ═══════════════════════════════════════════
#  Source: YouTube Search API
# ═══════════════════════════════════════════

def search_youtube(
    youtube_client,
    topic: str,
    watched_channel_ids: set[str],
    max_results: int = 5,
) -> list[dict]:
    """
    Search YouTube for videos on a topic from non-subscribed channels.

    Quota cost: 100 units per search.list call.
    """
    if not youtube_client:
        return []

    try:
        response = youtube_client.search().list(
            part="snippet",
            q=topic,
            type="video",
            order="date",
            maxResults=max_results,
            publishedAfter=None,  # will be filtered by dedup instead
        ).execute()
    except Exception as e:
        print(f"    YouTube Search error: {e}")
        return []

    results = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        channel_id = snippet.get("channelId", "")
        video_id = item["id"].get("videoId", "")

        # Skip videos from already-subscribed channels
        if channel_id in watched_channel_ids:
            continue

        results.append({
            "source": "youtube",
            "type": "video",
            "title": snippet.get("title", ""),
            "video_id": video_id,
            "channel": snippet.get("channelTitle", ""),
            "channel_id": channel_id,
            "url": f"https://youtube.com/watch?v={video_id}",
            "published": snippet.get("publishedAt", ""),
            "description": snippet.get("description", "")[:200],
        })

    return results


def get_video_view_count(youtube_client, video_id: str) -> int | None:
    """Fetch view count for a single video. Costs 1 quota unit."""
    if not youtube_client:
        return None
    try:
        response = youtube_client.videos().list(
            part="statistics",
            id=video_id,
        ).execute()
        items = response.get("items", [])
        if items:
            return int(items[0]["statistics"].get("viewCount", 0))
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════
#  Discovery Run — Main Orchestrator
# ═══════════════════════════════════════════

def run_discovery(
    vault_path: str,
    config: dict,
    db: QueueDB,
    topics_limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full discovery pipeline.

    1. Scan topics with discover_enabled: true
    2. Search each topic across all sources
    3. Deduplicate against queue.db + vault-index
    4. Auto-ingest qualifying YouTube videos
    5. Generate discovery report

    Returns summary dict.
    """
    discover_cfg = config.get("discover", {})
    if not discover_cfg.get("enabled", True):
        print("Discovery is disabled in config.yaml")
        return {"topics_scanned": 0, "results": 0}

    topics_per_run = topics_limit or discover_cfg.get("topics_per_run", 20)
    results_per_topic = discover_cfg.get("results_per_topic", 5)
    auto_ingest_yt = discover_cfg.get("auto_ingest_youtube", True)
    min_views = discover_cfg.get("auto_ingest_youtube_min_views", 10000)
    auto_add_articles = discover_cfg.get("auto_add_articles", False)
    auto_add_papers = discover_cfg.get("auto_add_papers", False)

    # 1. Scan topics
    all_topics = scan_discoverable_topics(vault_path)
    if not all_topics:
        print("No topics with discover_enabled: true found.")
        return {"topics_scanned": 0, "results": 0}

    topics = all_topics[:topics_per_run]
    print(f"Scanning {len(topics)} topic(s) "
          f"(of {len(all_topics)} enabled, limit {topics_per_run})...\n")

    # 2. Build dedup set
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    vault_index = load_existing_index(index_path) or {
        "notes": {"topics": [], "people": [], "channels": [], "videos": [], "articles": []},
        "topic_aliases": {},
    }
    known = build_known_set(db, vault_index)

    # 3. Initialize clients
    openalex = OpenAlexClient()
    crossref = CrossrefClient()

    brave = None
    try:
        brave = BraveSearchClient()
    except ValueError:
        print("  Brave API key not configured — skipping article search\n")

    # YouTube client (optional)
    youtube_client = None
    yt_api_key = config.get("youtube_api_key", "")
    if yt_api_key:
        try:
            from googleapiclient.discovery import build as yt_build
            youtube_client = yt_build("youtube", "v3", developerKey=yt_api_key)
        except Exception as e:
            print(f"  YouTube API unavailable: {e}\n")

    # Get watched channel IDs to filter them out of YouTube results
    watched_channels = db.get_watched_channels()
    watched_ids = {ch["channel_id"] for ch in watched_channels}

    # 4. Search each topic
    all_results = {}  # topic_concept -> list of results
    total_found = 0
    total_new = 0
    auto_queued = 0
    papers_for_review = []
    articles_for_review = []
    videos_for_review = []

    for topic in topics:
        concept = topic["concept"]
        print(f"  [{concept}]")
        topic_results = []
        seen_titles = set()  # local dedup within this topic

        # OpenAlex
        papers = search_openalex(openalex, concept, max_results=results_per_topic)
        for p in papers:
            key = (p.get("title") or "").lower().strip()
            if key and key not in seen_titles and not is_duplicate(p, known):
                seen_titles.add(key)
                p["topic"] = concept
                topic_results.append(p)

        # Crossref (supplement — fewer results to avoid overlap)
        cr_papers = search_crossref(crossref, concept, max_results=3)
        for p in cr_papers:
            key = (p.get("title") or "").lower().strip()
            if key and key not in seen_titles and not is_duplicate(p, known):
                seen_titles.add(key)
                p["topic"] = concept
                topic_results.append(p)

        # Brave articles
        if brave:
            articles = search_brave(brave, concept, max_results=results_per_topic)
            for a in articles:
                key = (a.get("title") or "").lower().strip()
                if key and key not in seen_titles and not is_duplicate(a, known):
                    seen_titles.add(key)
                    a["topic"] = concept
                    topic_results.append(a)

        # YouTube Search
        if youtube_client:
            videos = search_youtube(
                youtube_client, concept, watched_ids,
                max_results=results_per_topic,
            )
            for v in videos:
                if not is_duplicate(v, known):
                    v["topic"] = concept
                    topic_results.append(v)

        total_found += len(topic_results)

        # Categorize results and apply auto-ingest rules
        for item in topic_results:
            total_new += 1
            known.add((item.get("title") or "").lower().strip())

            if item["type"] == "video":
                video_id = item.get("video_id")
                if video_id:
                    known.add(video_id)

                # Check auto-ingest for YouTube videos
                if auto_ingest_yt and video_id and youtube_client and not dry_run:
                    views = get_video_view_count(youtube_client, video_id)
                    item["view_count"] = views
                    if views is not None and views >= min_views:
                        url = f"https://youtube.com/watch?v={video_id}"
                        if not db.exists(video_id):
                            db.add_to_queue(
                                video_id=video_id,
                                url=url,
                                channel=item.get("channel"),
                                source="discover",
                            )
                            item["auto_ingested"] = True
                            auto_queued += 1
                            print(f"    Auto-queued: {item['title'][:50]} "
                                  f"({views:,} views)")
                        else:
                            item["auto_ingested"] = False
                    else:
                        item["auto_ingested"] = False
                        videos_for_review.append(item)
                else:
                    item["auto_ingested"] = False
                    videos_for_review.append(item)

            elif item["type"] == "paper":
                if auto_add_papers and not dry_run:
                    item["auto_ingested"] = True
                else:
                    item["auto_ingested"] = False
                    papers_for_review.append(item)

            elif item["type"] == "article":
                if auto_add_articles and not dry_run:
                    item["auto_ingested"] = True
                else:
                    item["auto_ingested"] = False
                    articles_for_review.append(item)

        if topic_results:
            print(f"    Found {len(topic_results)} new result(s)")
        else:
            print(f"    No new results")

        all_results[concept] = topic_results

    # 5. Generate discovery report
    report_path = generate_report(
        vault_path=vault_path,
        topics=topics,
        results=all_results,
        papers_for_review=papers_for_review,
        articles_for_review=articles_for_review,
        videos_for_review=videos_for_review,
        auto_queued=auto_queued,
        dry_run=dry_run,
    )

    summary = {
        "topics_scanned": len(topics),
        "total_found": total_found,
        "total_new": total_new,
        "auto_queued": auto_queued,
        "papers_for_review": len(papers_for_review),
        "articles_for_review": len(articles_for_review),
        "videos_for_review": len(videos_for_review),
        "report": report_path,
    }

    print(f"\n{'='*60}")
    print(f"Discovery complete")
    print(f"  Topics scanned:     {summary['topics_scanned']}")
    print(f"  New results:        {summary['total_new']}")
    print(f"  Auto-queued videos: {summary['auto_queued']}")
    print(f"  Papers for review:  {summary['papers_for_review']}")
    print(f"  Articles for review:{summary['articles_for_review']}")
    print(f"  Videos for review:  {summary['videos_for_review']}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    return summary


# ═══════════════════════════════════════════
#  Discovery Report Generator
# ═══════════════════════════════════════════

def generate_report(
    vault_path: str,
    topics: list[dict],
    results: dict[str, list[dict]],
    papers_for_review: list[dict],
    articles_for_review: list[dict],
    videos_for_review: list[dict],
    auto_queued: int,
    dry_run: bool = False,
) -> str:
    """
    Generate a Discovery Report markdown file.

    Written to _system/discovery/YYYY-MM-DD.md.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_dir = os.path.join(vault_path, "_system", "discovery")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{today}.md")

    total_results = sum(len(v) for v in results.values())
    topic_names = [t["concept"] for t in topics]

    lines = [
        f"# Discovery Report — {today}",
        "",
        f"**Topics scanned:** {len(topics)}",
        f"**New results found:** {total_results}",
        f"**Auto-queued videos:** {auto_queued}",
        f"**Papers for review:** {len(papers_for_review)}",
        f"**Articles for review:** {len(articles_for_review)}",
        f"**Videos for review:** {len(videos_for_review)}",
        "",
        f"{'> DRY RUN — no videos were auto-queued' if dry_run else ''}",
        "",
        "---",
        "",
    ]

    # ── Papers for Manual Review ──
    lines.append("## Papers for Review")
    lines.append("")
    if papers_for_review:
        for p in papers_for_review:
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            year = p.get("year")
            year_str = f" ({year})" if year else ""
            url = p.get("url", "")
            url_str = f" — [Link]({url})" if url else ""
            topic = p.get("topic", "")
            lines.append(f"- [ ] **{p['title']}**{year_str}")
            lines.append(f"  - {author_str}{url_str}")
            lines.append(f"  - Source: {p['source']} | Topic: [[{topic}]]")
            lines.append("")
    else:
        lines.append("*No new papers found.*")
        lines.append("")

    # ── Articles for Manual Review ──
    lines.append("## Articles for Review")
    lines.append("")
    if articles_for_review:
        for a in articles_for_review:
            url = a.get("url", "")
            domain = a.get("domain", "")
            topic = a.get("topic", "")
            desc = a.get("description", "")[:120]
            lines.append(f"- [ ] [{a['title']}]({url})")
            lines.append(f"  - {domain} | Topic: [[{topic}]]")
            if desc:
                lines.append(f"  - {desc}")
            lines.append("")
    else:
        lines.append("*No new articles found.*")
        lines.append("")

    # ── Videos for Manual Review ──
    lines.append("## Videos for Review")
    lines.append("")
    if videos_for_review:
        for v in videos_for_review:
            channel = v.get("channel", "Unknown")
            topic = v.get("topic", "")
            url = v.get("url", "")
            views = v.get("view_count")
            view_str = f" | {views:,} views" if views else ""
            lines.append(f"- [ ] [{v['title']}]({url})")
            lines.append(f"  - Channel: {channel}{view_str} | Topic: [[{topic}]]")
            lines.append("")
    else:
        lines.append("*No new videos found (or all auto-queued).*")
        lines.append("")

    # ── Auto-Queued Videos ──
    lines.append("## Auto-Queued Videos")
    lines.append("")
    auto_items = [
        item
        for topic_results in results.values()
        for item in topic_results
        if item.get("auto_ingested") and item["type"] == "video"
    ]
    if auto_items:
        for v in auto_items:
            views = v.get("view_count")
            view_str = f" ({views:,} views)" if views else ""
            lines.append(f"- [x] [{v['title']}]({v['url']}){view_str}")
            lines.append(f"  - Channel: {v.get('channel', '')} | Topic: [[{v.get('topic', '')}]]")
            lines.append("")
    else:
        lines.append("*No videos auto-queued this run.*")
        lines.append("")

    # ── Per-Topic Breakdown ──
    lines.append("---")
    lines.append("")
    lines.append("## Results by Topic")
    lines.append("")
    for topic in topics:
        concept = topic["concept"]
        topic_results = results.get(concept, [])
        lines.append(f"### [[{concept}]]")
        lines.append("")
        if not topic_results:
            lines.append("*No new results.*")
            lines.append("")
            continue
        for item in topic_results:
            icon = {"paper": "📄", "article": "🌐", "video": "🎥"}.get(
                item["type"], "•"
            )
            status = "✅" if item.get("auto_ingested") else "⏳"
            lines.append(f"- {status} {icon} {item.get('title', 'Untitled')}")
        lines.append("")

    content = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    return report_path


# ═══════════════════════════════════════════
#  List Enabled Topics
# ═══════════════════════════════════════════

def list_enabled(vault_path: str) -> None:
    """Print topics that have discover_enabled: true."""
    topics = scan_discoverable_topics(vault_path)
    if not topics:
        print("No topics with discover_enabled: true.")
        return

    print(f"{'Topic':40s} {'Definition'}")
    print("-" * 80)
    for t in topics:
        definition = (t.get("definition") or "")[:40]
        print(f"{t['concept']:40s} {definition}")
    print(f"\n{len(topics)} topic(s) with discovery enabled")


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Web Topic Discovery System",
        epilog="""
Examples:
  python discover.py                    # Run discovery
  python discover.py --topics 5         # Limit to 5 topics
  python discover.py --dry-run          # Report only, no auto-ingest
  python discover.py --list-enabled     # Show discoverable topics
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=None,
        help="Max number of topics to scan (overrides config topics_per_run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate report without auto-ingesting any videos",
    )
    parser.add_argument(
        "--list-enabled",
        action="store_true",
        help="List topics with discover_enabled: true",
    )

    args = parser.parse_args()

    vault_path = get_vault_path()
    config = load_config()

    if args.list_enabled:
        list_enabled(vault_path)
        return

    db = get_db(vault_path)
    try:
        run_discovery(
            vault_path=vault_path,
            config=config,
            db=db,
            topics_limit=args.topics,
            dry_run=args.dry_run,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
