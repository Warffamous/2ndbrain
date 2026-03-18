#!/usr/bin/env python3
"""
watch.py — Channel Watch System

Monitors subscribed YouTube channels for new uploads and auto-triggers
the ingest pipeline.

Quota budget: ~120 YouTube API units/day for 50 channels.
  - check: 1 playlistItems.list per channel (1 unit each) × 2 checks/day = 100 units
  - subscribe overhead: channels.list (1 unit) per new subscription
  - Reserve of 200 units in QuotaGuard covers this

Usage:
    python watch.py --subscribe "https://youtube.com/@hubermanlab"
    python watch.py --unsubscribe UCxxxxxx
    python watch.py --list
    python watch.py --check-now
    python watch.py --check-now --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import yaml

from queue_db import QueueDB
from youtube_extractor import YouTubeExtractor, extract_channel_handle
from note_writer import atomic_write, find_existing_note, sanitize_filename


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
    return os.path.dirname(os.path.abspath(__file__))


def get_db(vault_path: str) -> QueueDB:
    db_path = os.path.join(vault_path, "_system", "queue.db")
    db = QueueDB(db_path)
    db.init_db()
    return db


# ═══════════════════════════════════════════
#  Channel Resolution
# ═══════════════════════════════════════════

def resolve_channel(
    extractor: YouTubeExtractor,
    url: str,
) -> dict:
    """
    Resolve a channel URL or handle to its ID, name, and metadata.

    Uses channels.list (1 quota unit). Returns dict with:
      channel_id, channel_name, description, subscriber_count, url
    """
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required")

    handle = extract_channel_handle(url)
    if handle:
        # Resolve @handle → channel ID + snippet in one call
        clean = handle.lstrip("@")
        response = extractor.youtube.channels().list(
            part="id,snippet,statistics",
            forHandle=clean,
        ).execute()
    else:
        # Assume bare channel ID or /channel/UCxxx URL
        channel_id = url.rstrip("/").split("/")[-1]
        response = extractor.youtube.channels().list(
            part="id,snippet,statistics",
            id=channel_id,
        ).execute()

    items = response.get("items", [])
    if not items:
        raise ValueError(f"Channel not found: {url}")

    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})

    return {
        "channel_id": item["id"],
        "channel_name": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "subscriber_count": stats.get("subscriberCount", "unknown"),
        "url": f"https://youtube.com/channel/{item['id']}",
    }


def get_uploads_playlist_id(channel_id: str) -> str:
    """Convert a channel ID to its uploads playlist ID."""
    if channel_id.startswith("UC"):
        return "UU" + channel_id[2:]
    raise ValueError(
        f"Cannot derive uploads playlist from channel ID: {channel_id}. "
        "Expected UC-prefixed ID."
    )


# ═══════════════════════════════════════════
#  Subscribe
# ═══════════════════════════════════════════

def subscribe(
    url: str,
    extractor: YouTubeExtractor,
    db: QueueDB,
    vault_path: str,
) -> bool:
    """
    Subscribe to a channel: resolve metadata, store in DB, create Channel note.

    Quota cost: 1 unit (channels.list with snippet+statistics).
    Returns True if newly subscribed, False if already subscribed.
    """
    print(f"Resolving channel: {url}")
    info = resolve_channel(extractor, url)
    channel_id = info["channel_id"]
    channel_name = info["channel_name"]

    print(f"  Channel: {channel_name} ({channel_id})")

    # Add to DB
    added = db.add_channel(channel_id, channel_name)
    if not added:
        print(f"  Already subscribed to {channel_name}")
        return False

    print(f"  Subscribed to {channel_name}")

    # Create Channel note if it doesn't exist
    channels_folder = os.path.join(vault_path, "Channels")
    safe_name = sanitize_filename(channel_name)
    existing = find_existing_note(channels_folder, safe_name)

    if not existing:
        filepath = os.path.join(channels_folder, f"{safe_name}.md")
        content = f"""---
type: channel
name: "{channel_name}"
url: "{info['url']}"
focus_areas:
- (none)
subscriber_count: "{info['subscriber_count']}"
credibility_notes: ""
ingested_count: 0
watched: true
---

## About

{info['description'][:500] if info['description'] else 'Subscribed via watch.py.'}

## Ingested Videos

*No videos ingested yet.*
"""
        atomic_write(filepath, content)
        print(f"  Created channel note: {safe_name}.md")
    else:
        # Update existing note to set watched: true
        print(f"  Channel note already exists: {os.path.basename(existing)}")
        _set_watched_flag(existing)

    return True


def _set_watched_flag(filepath: str) -> None:
    """Set watched: true in an existing channel note's frontmatter."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple regex replacement in frontmatter
    updated = re.sub(
        r"^(watched:\s*)false\s*$",
        r"\g<1>true",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if updated != content:
        atomic_write(filepath, updated)
        print(f"  Set watched: true on {os.path.basename(filepath)}")


# ═══════════════════════════════════════════
#  Unsubscribe
# ═══════════════════════════════════════════

def unsubscribe(channel_id: str, db: QueueDB) -> bool:
    """Remove a channel from the watch list."""
    removed = db.remove_channel(channel_id)
    if removed:
        print(f"Unsubscribed from {channel_id}")
    else:
        print(f"Channel not found in watch list: {channel_id}")
    return removed


# ═══════════════════════════════════════════
#  List
# ═══════════════════════════════════════════

def list_channels(db: QueueDB) -> None:
    """Print all watched channels."""
    channels = db.get_watched_channels()
    if not channels:
        print("No watched channels. Use --subscribe to add one.")
        return

    print(f"{'Channel ID':26s} {'Name':30s} {'Last Checked':22s} {'Last Video'}")
    print("-" * 95)
    for c in channels:
        last = c.get("last_checked") or "never"
        if len(last) > 22:
            last = last[:19]  # trim microseconds from ISO timestamp
        vid = c.get("last_video_id") or "—"
        print(f"{c['channel_id']:26s} {c['channel_name'][:30]:30s} {last:22s} {vid}")

    print(f"\n{len(channels)} channel(s) watched")


# ═══════════════════════════════════════════
#  Check Now — Scan channels for new uploads
# ═══════════════════════════════════════════

def check_channels(
    extractor: YouTubeExtractor,
    db: QueueDB,
    dry_run: bool = False,
) -> dict:
    """
    Check all subscribed channels for new uploads.

    For each channel:
      1. Fetch the latest 5 videos from its uploads playlist (1 quota unit)
      2. Compare with last_video_id stored in DB
      3. Queue any new videos with source='watch'

    Quota cost: 1 playlistItems.list per channel (1 unit each).
    For 50 channels = 50 units per check.

    Returns summary dict.
    """
    if not extractor.youtube:
        raise RuntimeError("YouTube API key required for --check-now")

    channels = db.get_watched_channels()
    if not channels:
        print("No watched channels. Use --subscribe to add one.")
        return {"checked": 0, "new_videos": 0}

    print(f"Checking {len(channels)} channel(s) for new uploads...\n")

    total_new = 0
    errors = 0

    for ch in channels:
        channel_id = ch["channel_id"]
        channel_name = ch["channel_name"]
        last_video_id = ch.get("last_video_id")

        try:
            uploads_playlist = get_uploads_playlist_id(channel_id)
        except ValueError as e:
            print(f"  [{channel_name}] Skipping — {e}")
            errors += 1
            continue

        try:
            # Fetch latest 5 uploads — 1 quota unit
            response = extractor.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=5,
            ).execute()
        except Exception as e:
            print(f"  [{channel_name}] API error: {e}")
            errors += 1
            continue

        items = response.get("items", [])
        if not items:
            print(f"  [{channel_name}] No uploads found")
            db.update_channel_check(channel_id)
            continue

        # Collect video IDs newer than last_video_id
        new_video_ids = []
        latest_id = items[0]["contentDetails"]["videoId"]

        if last_video_id is None:
            # First check — only take the most recent video to avoid flooding
            new_video_ids = [latest_id]
            print(f"  [{channel_name}] First check — seeding with latest video: {latest_id}")
        else:
            for item in items:
                vid = item["contentDetails"]["videoId"]
                if vid == last_video_id:
                    break
                new_video_ids.append(vid)

            if new_video_ids:
                print(f"  [{channel_name}] {len(new_video_ids)} new video(s) found")
            else:
                print(f"  [{channel_name}] Up to date")

        # Queue new videos
        queued = 0
        for vid in new_video_ids:
            if dry_run:
                print(f"    [DRY RUN] Would queue: {vid}")
                queued += 1
            else:
                video_url = f"https://youtube.com/watch?v={vid}"
                if not db.exists(vid):
                    db.add_to_queue(
                        video_id=vid,
                        url=video_url,
                        channel=channel_name,
                        source="watch",
                    )
                    print(f"    Queued: {vid}")
                    queued += 1
                else:
                    print(f"    Already in queue: {vid}")

        total_new += queued

        # Update last checked + last video ID
        if not dry_run:
            db.update_channel_check(channel_id, last_video_id=latest_id)

    summary = {
        "checked": len(channels),
        "new_videos": total_new,
        "errors": errors,
    }

    print(f"\nCheck complete: {len(channels)} channels checked, "
          f"{total_new} new video(s) {'would be ' if dry_run else ''}queued"
          + (f", {errors} errors" if errors else ""))

    return summary


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Web Channel Watch System",
        epilog="""
Examples:
  python watch.py --subscribe "https://youtube.com/@hubermanlab"
  python watch.py --subscribe "https://youtube.com/channel/UC2D2CMWXMOVWx7giW1n3LIg"
  python watch.py --unsubscribe UC2D2CMWXMOVWx7giW1n3LIg
  python watch.py --list
  python watch.py --check-now
  python watch.py --check-now --dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subscribe",
        metavar="URL",
        help="Subscribe to a YouTube channel (URL or @handle)",
    )
    parser.add_argument(
        "--unsubscribe",
        metavar="CHANNEL_ID",
        help="Unsubscribe from a channel by its channel ID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all watched channels",
    )
    parser.add_argument(
        "--check-now",
        action="store_true",
        help="Check all channels for new uploads and queue them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --check-now: show what would be queued without actually queueing",
    )

    args = parser.parse_args()

    if not any([args.subscribe, args.unsubscribe, args.list, args.check_now]):
        parser.print_help()
        sys.exit(1)

    vault_path = get_vault_path()
    config = load_config()
    db = get_db(vault_path)

    try:
        # Build extractor when needed (subscribe + check-now require YouTube API)
        extractor = None
        yt_api_key = config.get("youtube_api_key", "")

        if args.subscribe or args.check_now:
            if not yt_api_key:
                print("Error: YouTube API key required. Set youtube_api_key in _system/config.yaml",
                      file=sys.stderr)
                sys.exit(1)
            extractor = YouTubeExtractor(api_key=yt_api_key)

        # ── Subscribe ──
        if args.subscribe:
            subscribe(args.subscribe, extractor, db, vault_path)

        # ── Unsubscribe ──
        if args.unsubscribe:
            unsubscribe(args.unsubscribe, db)

        # ── List ──
        if args.list:
            list_channels(db)

        # ── Check Now ──
        if args.check_now:
            summary = check_channels(extractor, db, dry_run=args.dry_run)

            # If new videos were queued and not dry-run, suggest running ingest
            if summary["new_videos"] > 0 and not args.dry_run:
                print(f"\nRun 'python ingest.py' to process {summary['new_videos']} "
                      "new video(s) from the queue.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
