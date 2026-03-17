#!/usr/bin/env python3
"""
youtube_extractor.py — YouTube Metadata & Transcript Extraction

Pure data-fetching module. No LLM calls.

Two responsibilities:
  1. Fetch video metadata via YouTube Data API v3 (google-api-python-client)
  2. Fetch transcript via youtube-transcript-api, with Whisper fallback

Returns a clean dict with all fields needed for Video note frontmatter.

Standalone CLI:
    python youtube_extractor.py "https://youtube.com/watch?v=abc123"
    python youtube_extractor.py abc123 --transcript-only
    python youtube_extractor.py abc123 --metadata-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import yaml

# ── Lazy imports for optional heavy deps ──
_youtube_api = None
_transcript_api = None


def _get_youtube_service(api_key: str):
    """Lazy-build the YouTube Data API service."""
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=api_key)


def _get_transcript_api():
    """Lazy-import youtube_transcript_api."""
    from youtube_transcript_api import YouTubeTranscriptApi
    return YouTubeTranscriptApi


# ═══════════════════════════════════════════
#  URL Parsing
# ═══════════════════════════════════════════

def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    parsed = urlparse(url)

    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/v/")):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]

    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0]

    # Bare video ID (11 chars)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    return None


def extract_playlist_id(url: str) -> str | None:
    """Extract playlist ID from a YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        return qs.get("list", [None])[0]
    return None


def extract_channel_handle(url: str) -> str | None:
    """Extract channel handle from URL like youtube.com/@handle."""
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path.startswith("/@"):
            return parsed.path.split("/")[1]  # includes @
    return None


# ═══════════════════════════════════════════
#  Helper Parsers
# ═══════════════════════════════════════════

def parse_duration(iso_duration: str) -> int:
    """Convert ISO 8601 duration (PT1H23M45S) to minutes."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 60 + minutes + (1 if seconds >= 30 else 0)


def parse_chapters_from_description(description: str) -> list[dict]:
    """Extract chapter markers from video description."""
    chapters = []
    pattern = r"(?:^|\n)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)"
    matches = re.findall(pattern, description)

    for timestamp, title in matches:
        parts = timestamp.split(":")
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            seconds = int(parts[0]) * 60 + int(parts[1])
        else:
            continue
        chapters.append({
            "timestamp": timestamp,
            "seconds": seconds,
            "title": title.strip(),
        })

    return chapters


# ═══════════════════════════════════════════
#  Whisper Fallback
# ═══════════════════════════════════════════

def _download_audio(video_id: str, output_dir: str) -> str:
    """Download audio from a YouTube video using yt-dlp."""
    output_path = os.path.join(output_dir, f"{video_id}.wav")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", os.path.join(output_dir, f"{video_id}.%(ext)s"),
        "--no-playlist",
        f"https://youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    if not os.path.exists(output_path):
        # yt-dlp may have produced a different extension, find it
        for f in os.listdir(output_dir):
            if f.startswith(video_id):
                output_path = os.path.join(output_dir, f)
                break

    return output_path


def whisper_transcribe(
    video_id: str,
    model: str = "medium",
    device: str = "cpu",
) -> list[dict]:
    """
    Transcribe a video using OpenAI Whisper as fallback.
    Downloads audio via yt-dlp, then runs Whisper.
    Returns list of {text, start, duration} dicts matching
    youtube-transcript-api format.
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "Whisper fallback requires openai-whisper: pip install openai-whisper\n"
            "Also requires yt-dlp: pip install yt-dlp"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Downloading audio for {video_id}...")
        audio_path = _download_audio(video_id, tmpdir)

        print(f"  Transcribing with Whisper ({model}) on {device}...")
        whisper_model = whisper.load_model(model, device=device)
        result = whisper_model.transcribe(audio_path)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "duration": seg["end"] - seg["start"],
        })

    return segments


# ═══════════════════════════════════════════
#  Main Extractor Class
# ═══════════════════════════════════════════

class YouTubeExtractor:
    """Fetches video metadata and transcripts from YouTube."""

    def __init__(
        self,
        api_key: str,
        transcript_source: str = "youtube_first",
        whisper_model: str = "medium",
        whisper_device: str = "cpu",
    ):
        self.api_key = api_key
        self.transcript_source = transcript_source
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self._youtube = None

    @property
    def youtube(self):
        if self._youtube is None and self.api_key:
            self._youtube = _get_youtube_service(self.api_key)
        return self._youtube

    # ── Metadata ──

    def get_video_metadata(self, video_id: str) -> dict:
        """
        Fetch video metadata from YouTube Data API.
        Returns a normalized dict with all fields needed for Video note frontmatter.
        """
        if not self.youtube:
            raise RuntimeError("YouTube API key not configured")

        response = self.youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id,
        ).execute()

        items = response.get("items", [])
        if not items:
            raise ValueError(f"Video not found: {video_id}")

        item = items[0]
        snippet = item["snippet"]
        content_details = item["contentDetails"]
        statistics = item.get("statistics", {})

        published_raw = snippet.get("publishedAt", "")
        try:
            date_published = datetime.fromisoformat(
                published_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_published = ""

        description = snippet.get("description", "")
        chapters = parse_chapters_from_description(description)

        return {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": description,
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "date_published": date_published,
            "date_ingested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "duration_iso": content_details.get("duration", ""),
            "duration_minutes": parse_duration(content_details.get("duration", "")),
            "tags": snippet.get("tags", []),
            "chapters": chapters,
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "url": f"https://youtube.com/watch?v={video_id}",
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        }

    # ── Transcript ──

    def get_transcript(self, video_id: str) -> dict:
        """
        Fetch transcript for a video.
        Respects transcript_source config:
          - "youtube_first": try youtube-transcript-api, fall back to Whisper
          - "whisper_only": skip youtube-transcript-api, go straight to Whisper

        Returns dict:
          {
            "segments": [{"text": str, "start": float, "duration": float}, ...],
            "full_text": str,
            "timestamped_text": str,
            "source": "youtube" | "whisper",
          }
        """
        segments = None
        source = None

        if self.transcript_source != "whisper_only":
            # Try youtube-transcript-api first
            try:
                segments = self._fetch_youtube_transcript(video_id)
                source = "youtube"
            except Exception as e:
                print(f"  YouTube transcript unavailable: {e}")

        if segments is None:
            # Whisper fallback
            print(f"  Falling back to Whisper transcription...")
            try:
                segments = whisper_transcribe(
                    video_id,
                    model=self.whisper_model,
                    device=self.whisper_device,
                )
                source = "whisper"
            except Exception as e:
                raise RuntimeError(
                    f"No transcript available for {video_id}. "
                    f"YouTube transcript failed and Whisper fallback failed: {e}"
                )

        full_text = " ".join(seg["text"] for seg in segments)
        timestamped_text = self._format_timestamped(segments)

        return {
            "segments": segments,
            "full_text": full_text,
            "timestamped_text": timestamped_text,
            "source": source,
        }

    def _fetch_youtube_transcript(self, video_id: str) -> list[dict]:
        """Fetch transcript via youtube-transcript-api."""
        YTApi = _get_transcript_api()
        transcript_list = YTApi.list_transcripts(video_id)

        # Prefer manually created English, then auto-generated English, then anything
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                transcript = next(iter(transcript_list))

        return transcript.fetch()

    @staticmethod
    def _format_timestamped(segments: list[dict]) -> str:
        """Format segments as timestamped text for LLM processing."""
        lines = []
        for seg in segments:
            minutes = int(seg["start"] // 60)
            seconds = int(seg["start"] % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {seg['text']}")
        return "\n".join(lines)

    # ── Channel Info ──

    def get_channel_info(self, channel_id: str) -> dict:
        """Fetch basic channel information."""
        if not self.youtube:
            raise RuntimeError("YouTube API key not configured")

        response = self.youtube.channels().list(
            part="snippet,statistics",
            id=channel_id,
        ).execute()

        items = response.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {channel_id}")

        item = items[0]
        snippet = item["snippet"]
        stats = item.get("statistics", {})

        sub_count = int(stats.get("subscriberCount", 0))
        if sub_count >= 1_000_000:
            sub_display = f"{sub_count / 1_000_000:.1f}M"
        elif sub_count >= 1_000:
            sub_display = f"{sub_count / 1_000:.1f}K"
        else:
            sub_display = str(sub_count)

        return {
            "channel_id": channel_id,
            "name": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "custom_url": snippet.get("customUrl", ""),
            "subscriber_count": sub_display,
            "video_count": int(stats.get("videoCount", 0)),
            "url": f"https://youtube.com/channel/{channel_id}",
        }

    def resolve_channel_handle(self, handle: str) -> str | None:
        """Resolve a @handle to a channel ID."""
        if not self.youtube:
            return None
        clean_handle = handle.lstrip("@")
        response = self.youtube.channels().list(
            part="id",
            forHandle=clean_handle,
        ).execute()
        items = response.get("items", [])
        return items[0]["id"] if items else None

    # ── Combined Extraction ──

    def extract(self, video_id: str) -> dict:
        """
        Full extraction: metadata + transcript + channel info.
        Returns a single dict with everything downstream modules need.
        """
        print(f"Extracting data for video: {video_id}")

        print("  Fetching metadata...")
        metadata = self.get_video_metadata(video_id)

        print("  Fetching transcript...")
        transcript = self.get_transcript(video_id)

        print("  Fetching channel info...")
        channel_info = self.get_channel_info(metadata["channel_id"])

        return {
            "metadata": metadata,
            "transcript": transcript,
            "channel": channel_info,
        }


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def load_config() -> dict:
    """Load config.yaml from _system/ relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "_system", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Metadata & Transcript Extractor"
    )
    parser.add_argument(
        "url",
        help="YouTube video URL or video ID",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch metadata (no transcript)",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Only fetch transcript (no metadata)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="YouTube Data API key (overrides config.yaml)",
    )
    parser.add_argument(
        "--transcript-source",
        choices=["youtube_first", "whisper_only"],
        default=None,
        help="Transcript source strategy (overrides config.yaml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON",
    )

    args = parser.parse_args()

    # Parse video ID
    video_id = extract_video_id(args.url)
    if not video_id:
        print(f"Error: Could not extract video ID from: {args.url}", file=sys.stderr)
        sys.exit(1)

    print(f"Video ID: {video_id}")

    # Load config
    config = load_config()
    api_key = args.api_key or config.get("youtube_api_key", "")
    transcript_source = args.transcript_source or config.get("transcript_source", "youtube_first")
    whisper_model = config.get("whisper_model", "medium")
    whisper_device = config.get("whisper_device", "cpu")

    extractor = YouTubeExtractor(
        api_key=api_key,
        transcript_source=transcript_source,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
    )

    result = {}

    # Metadata
    if not args.transcript_only:
        if not api_key:
            print("Error: YouTube API key required for metadata. Set in config.yaml or --api-key", file=sys.stderr)
            sys.exit(1)
        metadata = extractor.get_video_metadata(video_id)
        result["metadata"] = metadata

        if not args.output_json:
            print(f"\n{'='*50}")
            print(f"Title:     {metadata['title']}")
            print(f"Channel:   {metadata['channel_title']}")
            print(f"Published: {metadata['date_published']}")
            print(f"Duration:  {metadata['duration_minutes']} min")
            print(f"Views:     {metadata['view_count']:,}")
            print(f"Tags:      {', '.join(metadata['tags'][:10])}")
            if metadata["chapters"]:
                print(f"Chapters:  {len(metadata['chapters'])}")
                for ch in metadata["chapters"][:5]:
                    print(f"  {ch['timestamp']} {ch['title']}")
                if len(metadata["chapters"]) > 5:
                    print(f"  ... and {len(metadata['chapters']) - 5} more")
            print(f"{'='*50}")

    # Transcript
    if not args.metadata_only:
        transcript = extractor.get_transcript(video_id)
        result["transcript"] = {
            "source": transcript["source"],
            "segment_count": len(transcript["segments"]),
            "char_count": len(transcript["full_text"]),
            "full_text": transcript["full_text"],
        }

        if not args.output_json:
            print(f"\nTranscript source: {transcript['source']}")
            print(f"Segments: {len(transcript['segments'])}")
            print(f"Characters: {len(transcript['full_text']):,}")
            print(f"\nFirst 500 chars:")
            print(transcript["full_text"][:500])
            print("...")

    # Channel info (only with full extraction)
    if not args.transcript_only and not args.metadata_only:
        try:
            channel_info = extractor.get_channel_info(result["metadata"]["channel_id"])
            result["channel"] = channel_info
            if not args.output_json:
                print(f"\nChannel: {channel_info['name']} ({channel_info['subscriber_count']} subscribers)")
        except Exception as e:
            print(f"  Warning: Could not fetch channel info: {e}", file=sys.stderr)

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
