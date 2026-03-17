#!/usr/bin/env python3
"""
note_writer.py — Vault Note Writer

Takes the structured output from llm_processor.py and writes notes to the vault.

Responsibilities:
  1. Render each note type (Video, Person, Channel, Topic) from Templates/
  2. Check vault-index.json before writing — update existing notes, don't duplicate
  3. Write files to correct vault folders per constitution naming conventions
  4. Atomic writes — write to temp file, then os.replace() — so crashes don't corrupt
  5. Call vault_index.update() after all writes to refresh the index

Usage:
    # Programmatic
    writer = NoteWriter(vault_path)
    result = writer.write_all(note_generation, extraction_data)

    # CLI (test with llm_processor JSON output)
    python note_writer.py /tmp/llm_output.json
    python note_writer.py /tmp/llm_output.json --dry-run
"""

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

import vault_index


# ═══════════════════════════════════════════
#  Filename Sanitization
# ═══════════════════════════════════════════

def sanitize_filename(name: str) -> str:
    """
    Clean a string for use as a filename.
    Rules from constitution:
      - No special characters except hyphens and spaces
      - No slashes, colons, quotes, etc.
    """
    # Replace common problematic characters
    cleaned = name.replace("/", "-").replace("\\", "-")
    cleaned = cleaned.replace(":", " -").replace('"', "").replace("'", "")
    cleaned = cleaned.replace("?", "").replace("*", "").replace("|", "")
    cleaned = cleaned.replace("<", "").replace(">", "")
    # Collapse multiple spaces/hyphens
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip(" -")


# ═══════════════════════════════════════════
#  YAML Helpers
# ═══════════════════════════════════════════

def format_yaml_list(items: list[str], wikilink: bool = False) -> str:
    """Format a list as YAML list items, optionally wrapping in [[wikilinks]]."""
    if not items:
        return "- (none)"
    lines = []
    for item in items:
        if wikilink:
            lines.append(f'- "[[{item}]]"')
        else:
            lines.append(f'- "{item}"')
    return "\n".join(lines)


def format_yaml_list_plain(items: list[str]) -> str:
    """Format a list as bare YAML items (no quotes, no links)."""
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


# ═══════════════════════════════════════════
#  Atomic File Write
# ═══════════════════════════════════════════

def atomic_write(filepath: str, content: str) -> None:
    """
    Write content to filepath atomically.
    Writes to a temp file in the same directory, then os.replace().
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp", prefix=".nw_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ═══════════════════════════════════════════
#  Existing Note Parsers (for updates)
# ═══════════════════════════════════════════

def read_existing_note(filepath: str) -> tuple[dict | None, str]:
    """
    Read an existing note file.
    Returns (frontmatter_dict, body_text) or (None, "") if not found.
    """
    if not os.path.exists(filepath):
        return None, ""

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not match:
        return None, content

    try:
        fm = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        fm = None

    return fm, match.group(2)


def find_existing_note(folder: str, name: str) -> str | None:
    """
    Find an existing note file by its stem name (case-insensitive).
    Returns the full path if found, None otherwise.
    """
    if not os.path.exists(folder):
        return None

    name_lower = name.lower()
    for filename in os.listdir(folder):
        if filename.endswith(".md") and Path(filename).stem.lower() == name_lower:
            return os.path.join(folder, filename)

    return None


# ═══════════════════════════════════════════
#  Note Renderers
# ═══════════════════════════════════════════

class NoteWriter:
    """Writes all note types to the vault with atomic file operations."""

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.folders = {
            "videos": os.path.join(vault_path, "Sources", "Videos"),
            "people": os.path.join(vault_path, "People"),
            "channels": os.path.join(vault_path, "Channels"),
            "topics": os.path.join(vault_path, "Topics"),
        }
        self.written_files: list[str] = []
        self.updated_files: list[str] = []
        self.skipped: list[str] = []

    # ── Video Note ──

    def write_video_note(
        self,
        video_note: dict,
        metadata: dict,
    ) -> str:
        """
        Render and write a Video note.
        Videos are always new (keyed by date + title), never updated.
        Returns the written filepath.
        """
        filename = sanitize_filename(video_note.get("filename", ""))
        if not filename:
            # Fallback: construct from metadata
            date = metadata.get("date_published", "unknown")
            title = sanitize_filename(video_note.get("cleaned_title", metadata.get("title", "Untitled")))
            filename = f"{date} - {title}"

        filepath = os.path.join(self.folders["videos"], f"{filename}.md")

        # Check if this exact video already exists
        if os.path.exists(filepath):
            print(f"  Video note already exists, skipping: {filename}")
            self.skipped.append(filepath)
            return filepath

        speakers_yaml = format_yaml_list(video_note.get("speakers", []), wikilink=True)
        topics_yaml = format_yaml_list(video_note.get("topics", []), wikilink=True)

        content = f"""---
type: video
title: "{video_note.get('cleaned_title', metadata.get('title', ''))}"
channel: "[[{metadata.get('channel_title', '')}]]"
url: "{metadata.get('url', '')}"
date_published: {metadata.get('date_published', '')}
date_ingested: {metadata.get('date_ingested', datetime.now(timezone.utc).strftime('%Y-%m-%d'))}
duration_minutes: {metadata.get('duration_minutes', 0)}
speakers:
{speakers_yaml}
topics:
{topics_yaml}
key_claims: {video_note.get('key_claims_count', 0)}
has_scholarly_refs: {str(video_note.get('has_scholarly_refs', False)).lower()}
status: complete
---

## Summary

{video_note.get('summary', '')}

## Key Claims

{video_note.get('claims_markdown', '*No claims extracted.*')}

## Notable Quotes

{video_note.get('quotes_markdown', '*No quotes extracted.*')}

## Chapters

{video_note.get('chapters_markdown', '*No chapters available.*')}

## Scholarly References

{video_note.get('scholarly_refs_markdown', '*No scholarly references found.*')}

## Related Notes

{video_note.get('related_notes_markdown', '')}
"""

        atomic_write(filepath, content)
        self.written_files.append(filepath)
        print(f"  Wrote video note: {filename}")
        return filepath

    # ── Person Note ──

    def write_person_note(
        self,
        person: dict,
        video_title: str,
    ) -> str:
        """
        Render and write or update a Person note.
        If person exists, update appearances count and add video to appearances list.
        Returns the written filepath.
        """
        name = person["name"]
        safe_name = sanitize_filename(name)
        filepath = os.path.join(self.folders["people"], f"{safe_name}.md")

        existing_path = find_existing_note(self.folders["people"], safe_name)

        if existing_path:
            return self._update_person_note(existing_path, person, video_title)

        # Create new person note
        affiliations_yaml = format_yaml_list(person.get("affiliations", []))
        channels_yaml = format_yaml_list(person.get("channels", []), wikilink=True)
        topics_yaml = format_yaml_list(person.get("topics", []), wikilink=True)
        conflicts_yaml = format_yaml_list_plain(person.get("conflicts", []))
        evidence_yaml = format_yaml_list_plain(person.get("evidence_links", []))

        content = f"""---
type: person
name: "{name}"
role: "{person.get('role', '')}"
affiliations:
{affiliations_yaml}
channels:
{channels_yaml}
topics:
{topics_yaml}
publications_count: {person.get('publications_count', 'unknown')}
h_index: {person.get('h_index', 'unknown')}
conflicts:
{conflicts_yaml}
evidence_links:
{evidence_yaml}
date_created: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
appearances: 1
---

## Background

{person.get('background', '')}

## Key Contributions

*To be populated as more content is ingested.*

## Appearances

- [[{video_title}]]

## Conflicts of Interest

{self._format_conflicts(person.get('conflicts', []))}
"""

        atomic_write(filepath, content)
        self.written_files.append(filepath)
        print(f"  Wrote new person note: {safe_name}")
        return filepath

    def _update_person_note(
        self,
        filepath: str,
        person: dict,
        video_title: str,
    ) -> str:
        """Update an existing person note: increment appearances, add video."""
        fm, body = read_existing_note(filepath)
        if fm is None:
            # Can't parse — skip update
            print(f"  Warning: Could not parse {filepath}, skipping update")
            self.skipped.append(filepath)
            return filepath

        # Increment appearances
        fm["appearances"] = fm.get("appearances", 0) + 1

        # Add new topics from this video (merge, no duplicates)
        existing_topics = fm.get("topics", [])
        # Extract plain topic names from existing wikilinks
        existing_topic_names = set()
        for t in existing_topics:
            if isinstance(t, str):
                clean = t.strip('"').strip("[[").strip("]]")
                existing_topic_names.add(clean.lower())

        new_topics = person.get("topics", [])
        for topic in new_topics:
            if topic.lower() not in existing_topic_names:
                existing_topics.append(f"[[{topic}]]")
        fm["topics"] = existing_topics

        # Add new conflicts if any
        new_conflicts = person.get("conflicts", [])
        existing_conflicts = fm.get("conflicts", [])
        if isinstance(existing_conflicts, list) and new_conflicts:
            for c in new_conflicts:
                if c not in existing_conflicts:
                    existing_conflicts.append(c)
            fm["conflicts"] = existing_conflicts

        # Rebuild note with updated frontmatter
        fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip()

        # Add video to appearances section in body
        appearance_link = f"- [[{video_title}]]"
        if "## Appearances" in body:
            # Insert the new appearance after the header
            body = body.replace(
                "## Appearances\n",
                f"## Appearances\n{appearance_link}\n",
                1,
            )
        else:
            body = body.rstrip() + f"\n\n## Appearances\n\n{appearance_link}\n"

        content = f"---\n{fm_str}\n---\n{body}"

        atomic_write(filepath, content)
        self.updated_files.append(filepath)
        print(f"  Updated person note: {Path(filepath).stem} (appearances: {fm['appearances']})")
        return filepath

    @staticmethod
    def _format_conflicts(conflicts: list[str]) -> str:
        if not conflicts:
            return "*No known conflicts of interest.*"
        return "\n".join(f"- {c}" for c in conflicts)

    # ── Channel Note ──

    def write_channel_note(
        self,
        channel_data: dict,
        channel_info: dict,
        video_title: str,
    ) -> str:
        """
        Render and write or update a Channel note.
        If channel exists, increment ingested_count and add video.
        """
        name = channel_data["name"]
        safe_name = sanitize_filename(name)
        filepath = os.path.join(self.folders["channels"], f"{safe_name}.md")

        existing_path = find_existing_note(self.folders["channels"], safe_name)

        if existing_path:
            return self._update_channel_note(existing_path, channel_data, video_title)

        # Create new channel note
        focus_yaml = format_yaml_list(channel_data.get("focus_areas", []), wikilink=True)
        channel_url = channel_info.get("url", "")
        subscriber_count = channel_info.get("subscriber_count", "unknown")

        content = f"""---
type: channel
name: "{name}"
url: "{channel_url}"
focus_areas:
{focus_yaml}
subscriber_count: "{subscriber_count}"
credibility_notes: "{channel_data.get('credibility_notes', '')}"
ingested_count: 1
watched: false
---

## About

{channel_data.get('about', '')}

## Ingested Videos

- [[{video_title}]]
"""

        atomic_write(filepath, content)
        self.written_files.append(filepath)
        print(f"  Wrote new channel note: {safe_name}")
        return filepath

    def _update_channel_note(
        self,
        filepath: str,
        channel_data: dict,
        video_title: str,
    ) -> str:
        """Update an existing channel note: increment ingested_count, add video."""
        fm, body = read_existing_note(filepath)
        if fm is None:
            print(f"  Warning: Could not parse {filepath}, skipping update")
            self.skipped.append(filepath)
            return filepath

        fm["ingested_count"] = fm.get("ingested_count", 0) + 1

        # Rebuild frontmatter
        fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip()

        # Add video to ingested videos section
        video_link = f"- [[{video_title}]]"
        if "## Ingested Videos" in body:
            body = body.replace(
                "## Ingested Videos\n",
                f"## Ingested Videos\n{video_link}\n",
                1,
            )
        else:
            body = body.rstrip() + f"\n\n## Ingested Videos\n\n{video_link}\n"

        content = f"---\n{fm_str}\n---\n{body}"

        atomic_write(filepath, content)
        self.updated_files.append(filepath)
        print(f"  Updated channel note: {Path(filepath).stem} (ingested: {fm['ingested_count']})")
        return filepath

    # ── Topic Note ──

    def write_topic_note(
        self,
        topic: dict,
        video_title: str,
    ) -> str:
        """
        Render and write a new Topic note.
        Only called for topics that don't exist yet.
        """
        concept = topic["concept"]
        safe_name = sanitize_filename(concept)
        filepath = os.path.join(self.folders["topics"], f"{safe_name}.md")

        # Double-check: don't overwrite existing
        existing_path = find_existing_note(self.folders["topics"], safe_name)
        if existing_path:
            print(f"  Topic note already exists, updating instead: {safe_name}")
            return self._update_topic_note(existing_path, video_title)

        related_yaml = format_yaml_list(topic.get("related_topics", []), wikilink=True)

        # Build aliases line for frontmatter
        aliases = topic.get("aliases", [])
        aliases_yaml = format_yaml_list_plain(aliases) if aliases else "[]"

        content = f"""---
type: topic
concept: "{concept}"
definition: "{topic.get('definition', '')}"
aliases:
{aliases_yaml}
related_topics:
{related_yaml}
sources_count: 1
discover_enabled: false
---

## Definition

{topic.get('definition', '')}

## Key Claims

*Claims will accumulate as more sources are ingested.*

## Sources

- [[{video_title}]]

## Related Topics

{self._format_related_topics(topic.get('related_topics', []))}
"""

        atomic_write(filepath, content)
        self.written_files.append(filepath)
        print(f"  Wrote new topic note: {safe_name}")
        return filepath

    def update_existing_topic(self, topic_name: str, video_title: str) -> str | None:
        """
        Update an existing topic note: increment sources_count, add video to Sources.
        Called for topics listed in topic_updates.
        """
        safe_name = sanitize_filename(topic_name)
        existing_path = find_existing_note(self.folders["topics"], safe_name)
        if not existing_path:
            print(f"  Warning: Topic '{topic_name}' listed for update but not found on disk")
            self.skipped.append(topic_name)
            return None

        return self._update_topic_note(existing_path, video_title)

    def _update_topic_note(self, filepath: str, video_title: str) -> str:
        """Increment sources_count and add video to Sources section."""
        fm, body = read_existing_note(filepath)
        if fm is None:
            print(f"  Warning: Could not parse {filepath}, skipping update")
            self.skipped.append(filepath)
            return filepath

        fm["sources_count"] = fm.get("sources_count", 0) + 1

        fm_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False).rstrip()

        video_link = f"- [[{video_title}]]"
        if "## Sources" in body:
            body = body.replace(
                "## Sources\n",
                f"## Sources\n{video_link}\n",
                1,
            )
        else:
            body = body.rstrip() + f"\n\n## Sources\n\n{video_link}\n"

        content = f"---\n{fm_str}\n---\n{body}"

        atomic_write(filepath, content)
        self.updated_files.append(filepath)
        print(f"  Updated topic note: {Path(filepath).stem} (sources: {fm['sources_count']})")
        return filepath

    @staticmethod
    def _format_related_topics(topics: list[str]) -> str:
        if not topics:
            return "*No related topics yet.*"
        return "\n".join(f"- [[{t}]]" for t in topics)

    # ── Orchestrator ──

    def write_all(
        self,
        note_generation: dict,
        extraction_data: dict,
    ) -> dict:
        """
        Write all notes from a single video's processing results.

        Args:
            note_generation: output from LLMProcessor.generate_notes()
              Keys: video_note, people, channel, topics, topic_updates
            extraction_data: output from YouTubeExtractor.extract()
              Keys: metadata, transcript, channel

        Returns:
            Summary dict with written/updated/skipped counts and paths.
        """
        metadata = extraction_data["metadata"]
        channel_info = extraction_data.get("channel", {})

        video_note = note_generation["video_note"]
        video_title = video_note.get("filename", "")
        if not video_title:
            date = metadata.get("date_published", "unknown")
            title = sanitize_filename(video_note.get("cleaned_title", metadata.get("title", "Untitled")))
            video_title = f"{date} - {title}"

        print(f"\nWriting notes for: {video_title}")
        print("-" * 50)

        # 1. Write video note
        self.write_video_note(video_note, metadata)

        # 2. Write/update person notes
        for person in note_generation.get("people", []):
            self.write_person_note(person, video_title)

        # 3. Write/update channel note
        channel_data = note_generation.get("channel", {})
        if channel_data:
            self.write_channel_note(channel_data, channel_info, video_title)

        # 4. Write new topic notes
        for topic in note_generation.get("topics", []):
            self.write_topic_note(topic, video_title)

        # 5. Update existing topic notes (increment sources_count)
        for topic_name in note_generation.get("topic_updates", []):
            self.update_existing_topic(topic_name, video_title)

        # 6. Update vault index
        print("\n  Updating vault index...")
        vault_index.update(self.vault_path)

        summary = {
            "written": len(self.written_files),
            "updated": len(self.updated_files),
            "skipped": len(self.skipped),
            "written_files": self.written_files[:],
            "updated_files": self.updated_files[:],
            "skipped_items": self.skipped[:],
        }

        print(f"\n  Done: {summary['written']} created, {summary['updated']} updated, {summary['skipped']} skipped")
        return summary


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def main():
    """
    Standalone CLI for testing note_writer with LLM processor JSON output.

    Usage:
        python note_writer.py /tmp/llm_output.json
        python note_writer.py /tmp/llm_output.json --dry-run
        python note_writer.py /tmp/llm_output.json --extraction /tmp/extracted.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Vault Note Writer — write processed notes to vault"
    )
    parser.add_argument(
        "input_json",
        help="Path to JSON from llm_processor.py (must contain note_generation key)",
    )
    parser.add_argument(
        "--extraction",
        default=None,
        help="Path to extraction JSON from youtube_extractor.py (for metadata). "
             "If not provided, uses minimal defaults.",
    )
    parser.add_argument(
        "--vault-path",
        default=None,
        help="Path to vault root (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without actually writing",
    )

    args = parser.parse_args()

    # Determine vault path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vault_path_arg = args.vault_path or script_dir

    # Load LLM output
    with open(args.input_json, "r") as f:
        llm_output = json.load(f)

    # note_generation may be nested under that key or be the top-level dict
    if "note_generation" in llm_output:
        note_generation = llm_output["note_generation"]
    else:
        note_generation = llm_output

    # Load extraction data
    if args.extraction:
        with open(args.extraction, "r") as f:
            extraction_data = json.load(f)
    else:
        # Provide minimal defaults so the writer can function
        extraction_data = {
            "metadata": {
                "title": note_generation.get("video_note", {}).get("cleaned_title", "Unknown"),
                "channel_title": note_generation.get("channel", {}).get("name", "Unknown"),
                "date_published": "",
                "date_ingested": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "duration_minutes": 0,
                "url": "",
            },
            "channel": {},
        }

    if args.dry_run:
        print("DRY RUN — would write the following notes:")
        print()

        vn = note_generation.get("video_note", {})
        print(f"  [VIDEO] {vn.get('filename', 'unknown')}")

        for p in note_generation.get("people", []):
            action = "CREATE" if p.get("is_new", True) else "UPDATE"
            print(f"  [PERSON {action}] {p['name']}")

        ch = note_generation.get("channel", {})
        if ch:
            action = "CREATE" if ch.get("is_new", True) else "UPDATE"
            print(f"  [CHANNEL {action}] {ch.get('name', 'unknown')}")

        for t in note_generation.get("topics", []):
            print(f"  [TOPIC CREATE] {t['concept']}")

        for t in note_generation.get("topic_updates", []):
            print(f"  [TOPIC UPDATE] {t}")

        return

    writer = NoteWriter(vault_path_arg)
    summary = writer.write_all(note_generation, extraction_data)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
