#!/usr/bin/env python3
"""
ingest.py — Knowledge-Web Ingest Pipeline Orchestrator

CLI entry point for the full ingest pipeline:
  extract (YouTube) → process (LLM) → write (vault notes)

Usage:
    # Single video
    python ingest.py "https://youtube.com/watch?v=abc123"

    # Playlist
    python ingest.py "https://youtube.com/playlist?list=PLxyz"

    # Entire channel (fetches all uploads)
    python ingest.py "https://youtube.com/@hubermanlab" --channel

    # Batch from file (one URL per line)
    python ingest.py --batch urls.txt

    # Retry failed
    python ingest.py --retry-failed

    # Status check
    python ingest.py --status
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import anthropic
import yaml

from queue_db import QueueDB
from youtube_extractor import (
    YouTubeExtractor,
    extract_channel_handle,
    extract_playlist_id,
    extract_video_id,
)
from enrichment import Enricher
from llm_processor import LLMProcessor
from note_writer import NoteWriter
import vault_index as vi


# ═══════════════════════════════════════════
#  Exponential Backoff Retry for Anthropic API
# ═══════════════════════════════════════════

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)
MAX_LLM_RETRIES = 5
BASE_DELAY = 1  # seconds


def _is_retryable(exc: Exception) -> bool:
    """Check if an Anthropic API exception is retryable."""
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


def _patch_llm_with_retry(processor: LLMProcessor, db: QueueDB | None = None) -> None:
    """
    Monkey-patch the LLMProcessor._call_llm method to wrap it
    with exponential backoff retry logic for 429/500/529 errors.
    Optionally tracks Anthropic API calls in the quota DB.
    """
    original_call = processor._call_llm

    def call_with_retry(prompt: str) -> str:
        last_exc = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                result = original_call(prompt)
                if db:
                    db.add_anthropic_call()
                return result
            except Exception as e:
                if not _is_retryable(e) or attempt == MAX_LLM_RETRIES:
                    raise
                last_exc = e
                delay = BASE_DELAY * (2 ** attempt)  # 1, 2, 4, 8, 16
                status = ""
                if isinstance(e, anthropic.APIStatusError):
                    status = f" (HTTP {e.status_code})"
                print(f"    Retryable error{status}: {e}")
                print(f"    Retrying in {delay}s (attempt {attempt + 1}/{MAX_LLM_RETRIES})...")
                time.sleep(delay)
        raise last_exc  # unreachable, but satisfies type checker

    processor._call_llm = call_with_retry


# ═══════════════════════════════════════════
#  YouTube API Quota Tracker
# ═══════════════════════════════════════════

# Per YouTube Data API v3 docs — cost in quota units per operation
QUOTA_COSTS = {
    "videos.list": 1,
    "channels.list": 1,
    "playlistItems.list": 1,
    "search.list": 100,
}


class QuotaGuard:
    """
    Tracks YouTube API quota usage and blocks requests when budget is exhausted.
    Constitution budget: 10,000 units/day.
    """

    def __init__(self, db: QueueDB, daily_limit: int = 10000, reserve: int = 200):
        self.db = db
        self.daily_limit = daily_limit
        self.reserve = reserve  # units reserved for Watch system

    @property
    def available(self) -> int:
        """Units available for Ingest (total minus reserve minus used)."""
        used = self.db.get_youtube_quota_today()
        return max(0, self.daily_limit - self.reserve - used)

    @property
    def used(self) -> int:
        return self.db.get_youtube_quota_today()

    def check(self, units_needed: int, operation: str = "") -> bool:
        """Check if we have enough quota. Returns True if OK."""
        if units_needed > self.available:
            op_str = f" for {operation}" if operation else ""
            print(f"  QUOTA WARNING: Need {units_needed} units{op_str}, "
                  f"only {self.available} available "
                  f"({self.used}/{self.daily_limit - self.reserve} used today)")
            return False
        return True

    def consume(self, units: int, operation: str = "") -> int:
        """Record quota consumption. Returns new daily total."""
        total = self.db.add_youtube_quota(units)
        return total

    def check_and_consume(self, operation: str, units: int | None = None) -> bool:
        """Check quota and consume if available. Returns True if consumed."""
        cost = units if units is not None else QUOTA_COSTS.get(operation, 1)
        if not self.check(cost, operation):
            return False
        self.consume(cost, operation)
        return True

    def estimate_video_cost(self) -> int:
        """Estimate quota cost for processing one video."""
        # videos.list (1) + channels.list (1) = 2 units minimum
        return 2

    def estimate_playlist_cost(self, video_count: int) -> int:
        """Estimate cost for resolving a playlist."""
        pages = (video_count + 49) // 50  # 50 items per page
        return pages  # playlistItems.list = 1 unit per page

    def print_status(self) -> None:
        """Print current quota status."""
        stats = self.db.get_quota_stats()
        avail = self.available
        pct = (self.used / (self.daily_limit - self.reserve) * 100) if self.daily_limit > self.reserve else 0
        print(f"  YouTube quota: {self.used}/{self.daily_limit - self.reserve} units "
              f"({pct:.0f}% used, {avail} available)")
        print(f"  Anthropic calls today: {stats.get('anthropic_calls', 0)}")
        print(f"  Videos processed today: {stats.get('videos_processed', 0)}")


# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════

def load_config() -> dict:
    """Load config.yaml from _system/ relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "_system", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_vault_path() -> str:
    """Get the vault path (directory containing this script)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_db(vault_path: str) -> QueueDB:
    """Create and initialize a QueueDB instance."""
    db_path = os.path.join(vault_path, "_system", "queue.db")
    db = QueueDB(db_path)
    db.init_db()
    return db


def load_vault_index_data(vault_path: str) -> dict:
    """Load vault-index.json."""
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return json.load(f)
    return {
        "last_updated": None,
        "notes": {"topics": [], "people": [], "channels": [], "videos": [], "articles": []},
        "topic_aliases": {},
    }


# ═══════════════════════════════════════════
#  URL Resolution — Playlist & Channel
# ═══════════════════════════════════════════

def resolve_playlist_videos(
    extractor: YouTubeExtractor,
    playlist_id: str,
    quota: QuotaGuard | None = None,
) -> list[str]:
    """
    Fetch all video IDs from a YouTube playlist.
    Paginates through all results (50 per page).
    Tracks quota: 1 unit per playlistItems.list call.
    """
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required for playlist resolution")

    video_ids = []
    next_page = None

    while True:
        if quota and not quota.check(1, "playlistItems.list"):
            print(f"  Stopping playlist resolution — quota exhausted (got {len(video_ids)} so far)")
            break

        request = extractor.youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page,
        )
        response = request.execute()

        if quota:
            quota.consume(1, "playlistItems.list")

        for item in response.get("items", []):
            vid = item["contentDetails"].get("videoId")
            if vid:
                video_ids.append(vid)

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return video_ids


def resolve_channel_uploads(
    extractor: YouTubeExtractor,
    channel_url: str,
    quota: QuotaGuard | None = None,
) -> list[str]:
    """
    Fetch all upload video IDs for a channel.
    Accepts a URL like https://youtube.com/@handle or a channel ID.
    Tracks quota: 1 unit for channels.list + N units for playlistItems.list.
    """
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required for channel resolution")

    # Resolve handle to channel ID if needed
    handle = extract_channel_handle(channel_url)
    if handle:
        if quota and not quota.check(1, "channels.list (handle resolve)"):
            raise RuntimeError("YouTube quota exhausted — cannot resolve channel handle")

        channel_id = extractor.resolve_channel_handle(handle)
        if quota:
            quota.consume(1, "channels.list")
        if not channel_id:
            raise ValueError(f"Could not resolve channel handle: {handle}")
    else:
        # Assume it's a channel URL with ID or bare ID
        channel_id = channel_url.rstrip("/").split("/")[-1]

    # Get the uploads playlist ID (replace UC prefix with UU)
    if channel_id.startswith("UC"):
        uploads_playlist = "UU" + channel_id[2:]
    else:
        if quota and not quota.check(1, "channels.list"):
            raise RuntimeError("YouTube quota exhausted — cannot look up channel")

        response = extractor.youtube.channels().list(
            part="contentDetails",
            id=channel_id,
        ).execute()

        if quota:
            quota.consume(1, "channels.list")

        items = response.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {channel_id}")
        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return resolve_playlist_videos(extractor, uploads_playlist, quota=quota)


# ═══════════════════════════════════════════
#  Queue Population
# ═══════════════════════════════════════════

def queue_single_video(db: QueueDB, url: str, source: str = "manual") -> int:
    """Parse a single video URL and add to queue. Returns count added."""
    video_id = extract_video_id(url)
    if not video_id:
        print(f"  Error: Could not extract video ID from: {url}", file=sys.stderr)
        return 0

    if db.exists(video_id):
        entry = db.get_entry(video_id)
        print(f"  Already in queue ({entry['status']}): {video_id}")
        return 0

    db.add_to_queue(video_id=video_id, url=url, source=source)
    print(f"  Queued: {video_id}")
    return 1


def queue_video_ids(db: QueueDB, video_ids: list[str], source: str = "manual") -> int:
    """Queue a list of video IDs. Returns count added."""
    added = 0
    for vid in video_ids:
        video_url = f"https://youtube.com/watch?v={vid}"
        if not db.exists(vid):
            db.add_to_queue(video_id=vid, url=video_url, source=source)
            added += 1
    return added


def queue_playlist(
    db: QueueDB,
    extractor: YouTubeExtractor,
    url: str,
    quota: QuotaGuard | None = None,
) -> int:
    """Resolve a playlist and queue all videos."""
    playlist_id = extract_playlist_id(url)
    if not playlist_id:
        print(f"  Error: Could not extract playlist ID from: {url}", file=sys.stderr)
        return 0

    print(f"  Resolving playlist: {playlist_id}")
    video_ids = resolve_playlist_videos(extractor, playlist_id, quota=quota)
    print(f"  Found {len(video_ids)} videos in playlist")

    added = queue_video_ids(db, video_ids, source="manual")
    skipped = len(video_ids) - added
    print(f"  Queued {added} new videos" + (f" ({skipped} already in queue)" if skipped else ""))
    return added


def queue_channel(
    db: QueueDB,
    extractor: YouTubeExtractor,
    url: str,
    quota: QuotaGuard | None = None,
) -> int:
    """Resolve a channel's uploads and queue all videos."""
    print(f"  Resolving channel uploads: {url}")
    video_ids = resolve_channel_uploads(extractor, url, quota=quota)
    print(f"  Found {len(video_ids)} uploads")

    added = queue_video_ids(db, video_ids, source="manual")
    skipped = len(video_ids) - added
    print(f"  Queued {added} new videos" + (f" ({skipped} already in queue)" if skipped else ""))
    return added


def queue_batch(
    db: QueueDB,
    filepath: str,
    extractor: YouTubeExtractor | None = None,
    quota: QuotaGuard | None = None,
) -> int:
    """
    Read URLs from a file (one per line) and queue them.
    Supports mixed URL types: video URLs, playlist URLs, channel URLs
    (channel URLs require a @handle or /channel/ prefix to be detected).
    """
    if not os.path.exists(filepath):
        print(f"Error: Batch file not found: {filepath}", file=sys.stderr)
        return 0

    with open(filepath, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Batch file: {len(urls)} URLs")
    added = 0
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url[:80]}")

        # Detect URL type
        if extract_playlist_id(url) and not extract_video_id(url):
            # Pure playlist URL (no video ID in it)
            if extractor and extractor.youtube:
                added += queue_playlist(db, extractor, url, quota=quota)
            else:
                print(f"  Skipping playlist (no YouTube API key): {url}")
        elif extract_channel_handle(url):
            if extractor and extractor.youtube:
                added += queue_channel(db, extractor, url, quota=quota)
            else:
                print(f"  Skipping channel (no YouTube API key): {url}")
        else:
            # Treat as single video URL
            added += queue_single_video(db, url, source="batch")

    print(f"\nBatch complete: {added} new videos queued from {len(urls)} URLs")
    return added


# ═══════════════════════════════════════════
#  Processing Pipeline
# ═══════════════════════════════════════════

def process_single_video(
    video_id: str,
    extractor: YouTubeExtractor,
    processor: LLMProcessor,
    vault_path: str,
    db: QueueDB,
    enricher: Enricher | None = None,
    quota: QuotaGuard | None = None,
    progress: str = "",
) -> bool:
    """
    Run the full pipeline for a single video:
      1. Extract (YouTube API + transcript)
      2. Process (3 LLM calls)
      3. Enrich (scholarly refs + person metadata)
      4. Write (vault notes)

    Returns True on success, False on failure.
    """
    progress_str = f" {progress}" if progress else ""
    print(f"\n{'='*60}")
    print(f"Processing{progress_str}: {video_id}")
    print(f"{'='*60}")

    # Pre-flight quota check: need ~2 units for extract (videos.list + channels.list)
    if quota and not quota.check(quota.estimate_video_cost(), "video extraction"):
        print("  Skipping — YouTube API quota exhausted for today")
        return False

    db.mark_processing(video_id)

    try:
        # Step 1: Extract
        print("\n[Step 1/4] Extracting YouTube data...")
        extraction_data = extractor.extract(video_id)

        # Track quota: videos.list (1) + channels.list (1) = 2 units
        if quota:
            quota.consume(2, "videos.list + channels.list")

        # Update queue with title/channel now that we have metadata
        title = extraction_data["metadata"]["title"]
        channel = extraction_data["metadata"]["channel_title"]
        conn = db._connect()
        conn.execute(
            "UPDATE queue SET title = ?, channel = ? WHERE id = ?",
            (title, channel, video_id),
        )
        conn.commit()

        print(f"  Title: {title}")
        print(f"  Channel: {channel}")
        print(f"  Duration: {extraction_data['metadata']['duration_minutes']} min")
        print(f"  Transcript: {extraction_data['transcript']['source']} "
              f"({len(extraction_data['transcript']['full_text']):,} chars)")

        # Step 2: LLM Processing
        print("\n[Step 2/4] LLM processing...")
        vault_index_data = load_vault_index_data(vault_path)
        llm_results = processor.process(extraction_data, vault_index_data)

        note_generation = llm_results["note_generation"]
        print(f"  Generated: video note + "
              f"{len(note_generation.get('people', []))} people + "
              f"{len(note_generation.get('topics', []))} new topics + "
              f"{len(note_generation.get('topic_updates', []))} topic updates")

        # Step 3: Enrich with scholarly refs + person metadata
        if enricher:
            print("\n[Step 3/4] Enriching with external data...")
            llm_results = enricher.enrich(llm_results)
            note_generation = llm_results["note_generation"]
        else:
            print("\n[Step 3/4] Enrichment skipped (not configured)")

        # Step 4: Write notes
        print("\n[Step 4/4] Writing notes to vault...")
        writer = NoteWriter(vault_path)
        writer.write_all(note_generation, extraction_data)

        # Mark complete + track stats
        db.mark_complete(video_id)
        db.add_video_processed()
        print(f"\n  Video {video_id} complete.")

        if quota:
            print(f"  Quota remaining: {quota.available} YouTube units")

        return True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n  FAILED: {error_msg}", file=sys.stderr)
        traceback.print_exc()
        db.mark_failed(video_id, error_msg[:500])
        return False


def process_queue(
    extractor: YouTubeExtractor,
    processor: LLMProcessor,
    vault_path: str,
    db: QueueDB,
    enricher: Enricher | None = None,
    quota: QuotaGuard | None = None,
    limit: int = 0,
    inter_video_delay: float = 2.0,
) -> dict:
    """
    Process all pending items in the queue.

    Args:
        limit: Max items to process (0 = unlimited)
        inter_video_delay: Seconds to wait between videos (rate limiting)

    Returns:
        Summary dict with success/failure counts.
    """
    pending = db.get_pending(limit=limit if limit > 0 else 1000)

    if not pending:
        print("No pending items in queue.")
        return {"processed": 0, "succeeded": 0, "failed": 0}

    total = len(pending)
    print(f"Processing {total} pending item(s)...\n")

    if quota:
        est_cost = total * quota.estimate_video_cost()
        if est_cost > quota.available:
            max_affordable = quota.available // quota.estimate_video_cost()
            print(f"  Warning: Estimated {est_cost} YouTube units needed, "
                  f"only {quota.available} available.")
            print(f"  Will process up to ~{max_affordable} videos before quota runs out.\n")

    succeeded = 0
    failed = 0
    skipped_quota = 0

    for i, entry in enumerate(pending):
        video_id = entry["id"]
        progress = f"[{i + 1}/{total}]"

        # Check quota before each video
        if quota and not quota.check(quota.estimate_video_cost(), "next video"):
            print(f"\n  Stopping — YouTube quota exhausted. "
                  f"{total - i} videos remaining in queue.")
            skipped_quota = total - i
            break

        ok = process_single_video(
            video_id, extractor, processor, vault_path, db,
            enricher=enricher, quota=quota, progress=progress,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1

        # Inter-video delay (rate limiting) — skip after last video
        if inter_video_delay > 0 and i < total - 1:
            time.sleep(inter_video_delay)

    summary = {
        "processed": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped_quota": skipped_quota,
    }

    print(f"\n{'='*60}")
    print(f"Queue processing complete: {succeeded} succeeded, {failed} failed", end="")
    if skipped_quota:
        print(f", {skipped_quota} skipped (quota)")
    else:
        print()
    print(f"{'='*60}")

    return summary


# ═══════════════════════════════════════════
#  Status Display
# ═══════════════════════════════════════════

def print_status(db: QueueDB, vault_path: str, quota: QuotaGuard | None = None) -> None:
    """Print comprehensive status: queue + quota + vault stats."""
    counts = db.get_status_counts()
    total = sum(counts.values())

    print("=" * 50)
    print("Knowledge-Web Ingest Status")
    print("=" * 50)
    print()
    print("Queue:")
    print(f"  Pending:    {counts.get('pending', 0):4d}")
    print(f"  Processing: {counts.get('processing', 0):4d}")
    print(f"  Complete:   {counts.get('complete', 0):4d}")
    print(f"  Failed:     {counts.get('failed', 0):4d}")
    print(f"  Total:      {total:4d}")
    print()

    # Quota stats
    if quota:
        print("Quota (today):")
        quota.print_status()
        print()

    # Show recent failures
    failed = db.list_all(status="failed")
    if failed:
        print("Recent failures:")
        for entry in failed[:5]:
            title = (entry.get("title") or entry["id"])[:50]
            error = (entry.get("error") or "unknown")[:60]
            retries = entry.get("retry_count", 0)
            print(f"  {title}")
            print(f"    Error: {error}")
            print(f"    Retries: {retries}")
        print()

    # Vault stats
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)

        print("Vault:")
        for note_type, titles in idx.get("notes", {}).items():
            print(f"  {note_type.capitalize():12s}: {len(titles):4d} notes")
        total_notes = sum(len(v) for v in idx.get("notes", {}).values())
        print(f"  {'Total':12s}: {total_notes:4d} notes")
        print(f"  Topic aliases: {len(idx.get('topic_aliases', {}))}")

    print("=" * 50)


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Web Ingest Pipeline",
        epilog="""
Examples:
  python ingest.py "https://youtube.com/watch?v=abc123"
  python ingest.py "https://youtube.com/playlist?list=PLxyz"
  python ingest.py "https://youtube.com/@hubermanlab" --channel
  python ingest.py --batch urls.txt
  python ingest.py --retry-failed
  python ingest.py --status
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="YouTube video URL, playlist URL, or channel URL",
    )
    parser.add_argument(
        "--channel",
        action="store_true",
        help="Treat the URL as a channel and ingest all uploads",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        default=None,
        help="Path to file with one URL per line",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset failed items to pending and reprocess",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue and vault status",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Queue items but don't process (useful for batch preview)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of videos to process from queue (0 = all)",
    )

    args = parser.parse_args()

    # Validate: need at least one action
    if not args.url and not args.batch and not args.retry_failed and not args.status:
        parser.print_help()
        sys.exit(1)

    vault_path = get_vault_path()
    config = load_config()
    db = get_db(vault_path)

    # Build quota guard
    quota = QuotaGuard(
        db=db,
        daily_limit=config.get("youtube_daily_quota", 10000),
        reserve=config.get("youtube_quota_reserve", 200),
    )

    max_retry_count = config.get("max_retry_count", 3)
    inter_video_delay = config.get("inter_video_delay", 2)

    try:
        # ── Status only ──
        if args.status:
            print_status(db, vault_path, quota=quota)
            if not args.url and not args.batch and not args.retry_failed:
                return

        # ── Build extractor (needed for playlist/channel resolution + processing) ──
        yt_api_key = config.get("youtube_api_key", "")
        extractor = YouTubeExtractor(
            api_key=yt_api_key,
            transcript_source=config.get("transcript_source", "youtube_first"),
            whisper_model=config.get("whisper_model", "medium"),
            whisper_device=config.get("whisper_device", "cpu"),
            webshare_proxy_username=config.get("webshare_proxy_username", ""),
            webshare_proxy_password=config.get("webshare_proxy_password", ""),
        )

        # ── Build LLM processor ──
        anthropic_key = config.get("anthropic_api_key", "")
        processor = None
        enricher = None
        if anthropic_key and not args.dry_run:
            processor = LLMProcessor(
                api_key=anthropic_key,
                model=config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=config.get("llm_max_tokens", 8000),
                temperature=config.get("llm_temperature", 0.2),
            )
            _patch_llm_with_retry(processor, db=db)

            # Build enricher (uses OpenAlex, Crossref, Brave)
            enricher = Enricher(config=config)

        # ── Queue population ──
        queued = 0

        if args.retry_failed:
            failed = db.get_failed(max_retries=max_retry_count)
            if not failed:
                print("No failed items eligible for retry "
                      f"(max_retry_count={max_retry_count}).")
            else:
                print(f"Resetting {len(failed)} failed item(s) to pending "
                      f"(retry_count < {max_retry_count})...")
                for entry in failed:
                    db.reset_for_retry(entry["id"])
                    queued += 1

        if args.batch:
            queued += queue_batch(
                db, args.batch,
                extractor=extractor if yt_api_key else None,
                quota=quota,
            )

        elif args.url:
            # Determine URL type
            if args.channel:
                if not yt_api_key:
                    print("Error: YouTube API key required for --channel. Set in config.yaml", file=sys.stderr)
                    sys.exit(1)
                queued += queue_channel(db, extractor, args.url, quota=quota)

            elif extract_playlist_id(args.url) and not extract_video_id(args.url):
                # Pure playlist URL (no video ID component)
                if not yt_api_key:
                    print("Error: YouTube API key required for playlist. Set in config.yaml", file=sys.stderr)
                    sys.exit(1)
                queued += queue_playlist(db, extractor, args.url, quota=quota)

            elif extract_playlist_id(args.url) and extract_video_id(args.url):
                # URL has both video and playlist — treat as playlist
                if yt_api_key:
                    queued += queue_playlist(db, extractor, args.url, quota=quota)
                else:
                    # Fallback: just queue the single video
                    print("  Note: Playlist detected but no YouTube API key — queueing single video only")
                    queued += queue_single_video(db, args.url)

            else:
                queued += queue_single_video(db, args.url)

        # ── Process queue ──
        if args.dry_run:
            print(f"\nDry run: {queued} item(s) queued. Use without --dry-run to process.")
            return

        if not processor:
            if queued > 0:
                print(f"\n{queued} item(s) queued. Anthropic API key not set — skipping processing.")
                print("Set anthropic_api_key in _system/config.yaml to enable processing.")
            return

        if not yt_api_key:
            print("Error: YouTube API key required for processing. Set in config.yaml", file=sys.stderr)
            sys.exit(1)

        # Process everything pending
        process_queue(
            extractor, processor, vault_path, db,
            enricher=enricher,
            quota=quota,
            limit=args.limit,
            inter_video_delay=inter_video_delay,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
