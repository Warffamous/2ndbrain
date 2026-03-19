#!/usr/bin/env python3
"""
vault_index.py — Knowledge-Web Vault Index Builder

Scans the vault folder structure and generates _system/vault-index.json,
which serves as the system's memory of all existing notes.

Usage:
    python vault_index.py --rebuild    # Full scan, regenerate from scratch
    python vault_index.py --update     # Incremental update (new/changed files)
    python vault_index.py --stats      # Print vault statistics
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(vault_path: str) -> dict:
    """Load configuration from _system/config.yaml."""
    config_path = os.path.join(vault_path, "_system", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_frontmatter(filepath: str) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return None

    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def get_note_title(filepath: str) -> str:
    """Extract note title from filename (without extension)."""
    return Path(filepath).stem


def scan_folder(folder_path: str) -> list[dict]:
    """Scan a folder for .md files and extract their metadata."""
    notes = []
    if not os.path.exists(folder_path):
        return notes

    for filename in os.listdir(folder_path):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        title = get_note_title(filepath)
        frontmatter = parse_frontmatter(filepath)
        mtime = os.path.getmtime(filepath)

        note_info = {
            "title": title,
            "file": filename,
            "mtime": mtime,
        }

        if frontmatter:
            note_info["frontmatter"] = frontmatter

        notes.append(note_info)

    return notes


def build_index(vault_path: str) -> dict:
    """Build the complete vault index by scanning all folders."""
    folder_map = {
        "topics": os.path.join(vault_path, "Topics"),
        "people": os.path.join(vault_path, "People"),
        "channels": os.path.join(vault_path, "Channels"),
        "videos": os.path.join(vault_path, "Sources", "Videos"),
        "articles": os.path.join(vault_path, "Sources", "Articles"),
    }

    index = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "notes": {},
        "topic_aliases": {},
    }

    for note_type, folder in folder_map.items():
        notes = scan_folder(folder)
        # Store just the titles in the main list (as per spec)
        index["notes"][note_type] = [n["title"] for n in notes]

        # Extract topic aliases if present
        if note_type == "topics":
            for note in notes:
                fm = note.get("frontmatter", {})
                if fm and "aliases" in fm:
                    aliases = fm["aliases"]
                    if isinstance(aliases, list):
                        for alias in aliases:
                            index["topic_aliases"][alias.lower()] = note["title"]

    return index


def load_existing_index(index_path: str) -> dict | None:
    """Load the existing vault index if it exists."""
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_index(index: dict, index_path: str) -> None:
    """Save the vault index to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def rebuild(vault_path: str) -> dict:
    """Full rebuild of the vault index."""
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    print("Rebuilding vault index from scratch...")

    index = build_index(vault_path)
    save_index(index, index_path)

    total = sum(len(v) for v in index["notes"].values())
    print(f"Index rebuilt: {total} notes indexed across {len(index['notes'])} categories.")
    print(f"Topic aliases: {len(index['topic_aliases'])}")
    print(f"Saved to: {index_path}")

    return index


def update(vault_path: str) -> dict:
    """Incremental update — rebuild and merge with existing aliases."""
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    existing = load_existing_index(index_path)

    print("Updating vault index...")

    index = build_index(vault_path)

    # Preserve manually added aliases from existing index
    if existing and "topic_aliases" in existing:
        for alias, target in existing["topic_aliases"].items():
            if alias not in index["topic_aliases"]:
                # Keep alias if the target topic still exists
                if target in index["notes"].get("topics", []):
                    index["topic_aliases"][alias] = target

    save_index(index, index_path)

    total = sum(len(v) for v in index["notes"].values())
    print(f"Index updated: {total} notes indexed.")
    print(f"Topic aliases: {len(index['topic_aliases'])}")

    return index


def print_stats(vault_path: str) -> None:
    """Print vault statistics."""
    index_path = os.path.join(vault_path, "_system", "vault-index.json")
    index = load_existing_index(index_path)

    if not index:
        print("No vault index found. Run --rebuild first.")
        return

    print("=" * 40)
    print("Knowledge-Web Vault Statistics")
    print("=" * 40)
    print(f"Last updated: {index.get('last_updated', 'unknown')}")
    print()

    for note_type, titles in index.get("notes", {}).items():
        print(f"  {note_type.capitalize():12s}: {len(titles):4d} notes")

    total = sum(len(v) for v in index.get("notes", {}).values())
    print(f"  {'Total':12s}: {total:4d} notes")
    print()
    print(f"  Topic aliases: {len(index.get('topic_aliases', {}))}")
    print("=" * 40)


def resolve_topic(query: str, index: dict) -> str | None:
    """
    Resolve a topic query against the vault index.
    Checks exact match first, then aliases.
    Returns the canonical topic name or None.
    """
    topics = index.get("notes", {}).get("topics", [])

    # Exact match
    if query in topics:
        return query

    # Case-insensitive match
    query_lower = query.lower()
    for topic in topics:
        if topic.lower() == query_lower:
            return topic

    # Alias match
    aliases = index.get("topic_aliases", {})
    if query_lower in aliases:
        return aliases[query_lower]

    return None


def note_exists(name: str, note_type: str, index: dict) -> bool:
    """Check if a note of a given type exists in the index."""
    notes = index.get("notes", {}).get(note_type, [])
    name_lower = name.lower()
    return any(n.lower() == name_lower for n in notes)


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge-Web Vault Index Builder"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Full scan, regenerate index from scratch",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incremental update (new/changed files)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print vault statistics",
    )
    parser.add_argument(
        "--vault-path",
        default=None,
        help="Path to the vault (overrides config)",
    )

    args = parser.parse_args()

    # Determine vault path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vault_path = args.vault_path or script_dir

    if not any([args.rebuild, args.update, args.stats]):
        parser.print_help()
        sys.exit(1)

    if args.rebuild:
        rebuild(vault_path)
    elif args.update:
        update(vault_path)

    if args.stats:
        print_stats(vault_path)


if __name__ == "__main__":
    main()
