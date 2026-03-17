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
from llm_processor import LLMProcessor
from note_writer import NoteWriter
import vault_index as vi


# ═══════════════════════════════════════════
#  Exponential Backoff Retry for Anthropic API
# ═══════════════════════════════════════════

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 529)
MAX_RETRIES = 5
BASE_DELAY = 1  # seconds


def _is_retryable(exc: Exception) -> bool:
    """Check if an Anthropic API exception is retryable."""
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


def _patch_llm_with_retry(processor: LLMProcessor) -> None:
    """
    Monkey-patch the LLMProcessor._call_llm method to wrap it
    with exponential backoff retry logic for 429/500/529 errors.
    """
    original_call = processor._call_llm

    def call_with_retry(prompt: str) -> str:
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return original_call(prompt)
            except Exception as e:
                if not _is_retryable(e) or attempt == MAX_RETRIES:
                    raise
                last_exc = e
                delay = BASE_DELAY * (2 ** attempt)  # 1, 2, 4, 8, 16
                status = ""
                if isinstance(e, anthropic.APIStatusError):
                    status = f" (HTTP {e.status_code})"
                print(f"    Retryable error{status}: {e}")
                print(f"    Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
        raise last_exc  # unreachable, but satisfies type checker

    processor._call_llm = call_with_retry


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

def resolve_playlist_videos(extractor: YouTubeExtractor, playlist_id: str) -> list[str]:
    """Fetch all video IDs from a YouTube playlist."""
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required for playlist resolution")

    video_ids = []
    next_page = None

    while True:
        request = extractor.youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page,
        )
        response = request.execute()

        for item in response.get("items", []):
            vid = item["contentDetails"].get("videoId")
            if vid:
                video_ids.append(vid)

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return video_ids


def resolve_channel_uploads(extractor: YouTubeExtractor, channel_url: str) -> list[str]:
    """
    Fetch all upload video IDs for a channel.
    Accepts a URL like https://youtube.com/@handle or a channel ID.
    """
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required for channel resolution")

    # Resolve handle to channel ID if needed
    handle = extract_channel_handle(channel_url)
    if handle:
        channel_id = extractor.resolve_channel_handle(handle)
        if not channel_id:
            raise ValueError(f"Could not resolve channel handle: {handle}")
    else:
        # Assume it's a channel URL with ID or bare ID
        channel_id = channel_url.rstrip("/").split("/")[-1]

    # Get the uploads playlist ID (replace UC prefix with UU)
    if channel_id.startswith("UC"):
        uploads_playlist = "UU" + channel_id[2:]
    else:
        # Try fetching channel to get uploads playlist
        response = extractor.youtube.channels().list(
            part="contentDetails",
            id=channel_id,
        ).execute()
        items = response.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {channel_id}")
        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    return resolve_playlist_videos(extractor, uploads_playlist)


# ═══════════════════════════════════════════
#  Queue Population
# ═══════════════════════════════════════════

def queue_single_video(db: QueueDB, url: str, source: str = "manual") -> int:
    """Parse a single video URL and add to queue. Returns count added."""
    video_id = extract_video_id(url)
    if not video_id:
        print(f"Error: Could not extract video ID from: {url}", file=sys.stderr)
        return 0

    if db.exists(video_id):
        entry = db.get_entry(video_id)
        print(f"  Already in queue ({entry['status']}): {video_id}")
        return 0

    db.add_to_queue(video_id=video_id, url=url, source=source)
    print(f"  Queued: {video_id}")
    return 1


def queue_playlist(db: QueueDB, extractor: YouTubeExtractor, url: str) -> int:
    """Resolve a playlist and queue all videos."""
    playlist_id = extract_playlist_id(url)
    if not playlist_id:
        print(f"Error: Could not extract playlist ID from: {url}", file=sys.stderr)
        return 0

    print(f"Resolving playlist: {playlist_id}")
    video_ids = resolve_playlist_videos(extractor, playlist_id)
    print(f"  Found {len(video_ids)} videos")

    added = 0
    for vid in video_ids:
        video_url = f"https://youtube.com/watch?v={vid}"
        if not db.exists(vid):
            db.add_to_queue(video_id=vid, url=video_url, source="manual")
            added += 1

    print(f"  Queued {added} new videos ({len(video_ids) - added} already in queue)")
    return added


def queue_channel(db: QueueDB, extractor: YouTubeExtractor, url: str) -> int:
    """Resolve a channel's uploads and queue all videos."""
    print(f"Resolving channel uploads: {url}")
    video_ids = resolve_channel_uploads(extractor, url)
    print(f"  Found {len(video_ids)} uploads")

    added = 0
    for vid in video_ids:
        video_url = f"https://youtube.com/watch?v={vid}"
        if not db.exists(vid):
            db.add_to_queue(video_id=vid, url=video_url, source="manual")
            added += 1

    print(f"  Queued {added} new videos ({len(video_ids) - added} already in queue)")
    return added


def queue_batch(db: QueueDB, filepath: str) -> int:
    """Read URLs from a file (one per line) and queue them."""
    if not os.path.exists(filepath):
        print(f"Error: Batch file not found: {filepath}", file=sys.stderr)
        return 0

    with open(filepath, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Batch file: {len(urls)} URLs")
    added = 0
    for url in urls:
        added += queue_single_video(db, url, source="batch")

    print(f"  Queued {added} new videos")
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
) -> bool:
    """
    Run the full pipeline for a single video:
      1. Extract (YouTube API + transcript)
      2. Process (3 LLM calls)
      3. Write (vault notes)

    Returns True on success, False on failure.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"{'='*60}")

    db.mark_processing(video_id)

    try:
        # Step 1: Extract
        print("\n[Step 1/3] Extracting YouTube data...")
        extraction_data = extractor.extract(video_id)

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
        print("\n[Step 2/3] LLM processing...")
        vault_index_data = load_vault_index_data(vault_path)
        llm_results = processor.process(extraction_data, vault_index_data)

        note_generation = llm_results["note_generation"]
        vn = note_generation.get("video_note", {})
        print(f"  Generated: video note + "
              f"{len(note_generation.get('people', []))} people + "
              f"{len(note_generation.get('topics', []))} new topics + "
              f"{len(note_generation.get('topic_updates', []))} topic updates")

        # Step 3: Write notes
        print("\n[Step 3/3] Writing notes to vault...")
        writer = NoteWriter(vault_path)
        write_summary = writer.write_all(note_generation, extraction_data)

        # Mark complete
        db.mark_complete(video_id)
        print(f"\n  Video {video_id} complete.")
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
    limit: int = 0,
) -> dict:
    """
    Process all pending items in the queue.

    Args:
        limit: Max items to process (0 = unlimited)

    Returns:
        Summary dict with success/failure counts.
    """
    pending = db.get_pending(limit=limit if limit > 0 else 1000)

    if not pending:
        print("No pending items in queue.")
        return {"processed": 0, "succeeded": 0, "failed": 0}

    print(f"Processing {len(pending)} pending item(s)...\n")

    succeeded = 0
    failed = 0

    for entry in pending:
        video_id = entry["id"]
        ok = process_single_video(video_id, extractor, processor, vault_path, db)
        if ok:
            succeeded += 1
        else:
            failed += 1

    summary = {
        "processed": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
    }

    print(f"\n{'='*60}")
    print(f"Queue processing complete: {succeeded} succeeded, {failed} failed")
    print(f"{'='*60}")

    return summary


# ═══════════════════════════════════════════
#  Status Display
# ═══════════════════════════════════════════

def print_status(db: QueueDB, vault_path: str) -> None:
    """Print comprehensive status: queue + vault stats."""
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

    # Show recent failures
    failed = db.list_all(status="failed")
    if failed:
        print("Recent failures:")
        for entry in failed[:5]:
            title = (entry.get("title") or entry["id"])[:50]
            error = (entry.get("error") or "unknown")[:60]
            print(f"  {title}")
            print(f"    Error: {error}")
            print(f"    Retries: {entry.get('retry_count', 0)}")
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

    try:
        # ── Status only ──
        if args.status:
            print_status(db, vault_path)
            if not args.url and not args.batch and not args.retry_failed:
                return

        # ── Build extractor (needed for playlist/channel resolution + processing) ──
        yt_api_key = config.get("youtube_api_key", "")
        extractor = YouTubeExtractor(
            api_key=yt_api_key,
            transcript_source=config.get("transcript_source", "youtube_first"),
            whisper_model=config.get("whisper_model", "medium"),
            whisper_device=config.get("whisper_device", "cpu"),
        )

        # ── Build LLM processor ──
        anthropic_key = config.get("anthropic_api_key", "")
        processor = None
        if anthropic_key and not args.dry_run:
            processor = LLMProcessor(
                api_key=anthropic_key,
                model=config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=config.get("llm_max_tokens", 8000),
                temperature=config.get("llm_temperature", 0.2),
            )
            _patch_llm_with_retry(processor)

        # ── Queue population ──
        queued = 0

        if args.retry_failed:
            failed = db.get_failed(max_retries=10)
            if not failed:
                print("No failed items to retry.")
            else:
                print(f"Resetting {len(failed)} failed item(s) to pending...")
                for entry in failed:
                    db.reset_for_retry(entry["id"])
                    queued += 1

        if args.batch:
            queued += queue_batch(db, args.batch)

        elif args.url:
            # Determine URL type
            if args.channel:
                if not yt_api_key:
                    print("Error: YouTube API key required for --channel. Set in config.yaml", file=sys.stderr)
                    sys.exit(1)
                queued += queue_channel(db, extractor, args.url)

            elif extract_playlist_id(args.url):
                if not yt_api_key:
                    print("Error: YouTube API key required for playlist. Set in config.yaml", file=sys.stderr)
                    sys.exit(1)
                queued += queue_playlist(db, extractor, args.url)

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
        process_queue(extractor, processor, vault_path, db, limit=args.limit)

    finally:
        db.close()


if __name__ == "__main__":
    main()
