#!/usr/bin/env python3
"""
llm_processor.py — LLM Processing Pipeline

Takes the raw extraction dict from youtube_extractor.py and runs three
sequential LLM calls via the Anthropic SDK:

  1. mine_description()  — Extract speakers, links, sponsors, chapters
                           from the video description
  2. process_transcript() — Clean transcript, extract key claims with
                           timestamps, topic tags, map speaker labels
  3. generate_notes()     — Vault-aware note content generation using
                           vault-index.json as context

Each call is its own method with prompt separated from calling logic.
Uses claude-sonnet-4-20250514 at temperature 0.2 (per config).
"""

import json
import os
import sys

import anthropic
import yaml


# ═══════════════════════════════════════════
#  Prompts (separated from calling logic)
# ═══════════════════════════════════════════

DESCRIPTION_MINING_PROMPT = """\
You are a structured data extractor. Analyze this YouTube video description and extract all structured information.

## Video Title
{title}

## Channel
{channel}

## Video Description
{description}

## Instructions

Extract the following and return as JSON:

1. **speakers**: List of people who appear in/are discussed in the video. For each:
   - "name": Full name (clean and standardized)
   - "role": Their stated role/title if mentioned, otherwise null
   - "is_host": true if they are the channel host, false if guest

2. **links**: All URLs found in the description. For each:
   - "url": The URL
   - "type": one of "paper", "article", "book", "product", "social", "sponsor", "other"
   - "label": Brief description of what it links to

3. **sponsors**: Companies/products that are sponsoring or being promoted. For each:
   - "name": Sponsor name
   - "relationship": How they're related (e.g., "sponsor", "affiliate", "own product")

4. **chapters**: Chapter markers if present in the description but NOT already captured by the API. For each:
   - "timestamp": The timestamp string (e.g., "5:30")
   - "title": Chapter title

5. **book_references**: Books or papers explicitly mentioned. For each:
   - "title": The title
   - "authors": Author name(s) if mentioned
   - "type": "book" or "paper"

6. **conflict_signals**: Any indicators of conflicts of interest (supplement sales, affiliate links, product promotions by the host, etc.)

Return ONLY valid JSON with these exact keys: speakers, links, sponsors, chapters, book_references, conflict_signals.
If a category has no entries, use an empty list [].\
"""

TRANSCRIPT_PROCESSING_PROMPT = """\
You are a knowledge extraction specialist. Process this video transcript to extract structured knowledge.

## Video Title
{title}

## Channel
{channel}

## Duration
{duration_minutes} minutes

## Known Speakers (from description analysis)
{known_speakers_json}

## Chapters
{chapters_json}

## Timestamped Transcript
{transcript}

## Instructions

Analyze the full transcript and extract:

1. **cleaned_summary**: A 3-5 sentence summary of the video's core content and arguments. Be specific and substantive — no filler like "In this video, the host discusses...". Lead with the actual content.

2. **key_claims**: The 5-15 most important factual claims, arguments, or insights made. For each:
   - "claim": The claim stated clearly and precisely (1-2 sentences)
   - "timestamp": Approximate timestamp where this claim is made (e.g., "12:30")
   - "speaker": Who made the claim (use full name from known speakers if possible)
   - "confidence": "stated_as_fact", "hypothesis", "opinion", or "citing_research"

3. **notable_quotes**: 3-8 particularly quotable or important statements (verbatim or near-verbatim, < 2 sentences each). For each:
   - "quote": The quote text
   - "speaker": Who said it
   - "timestamp": Approximate timestamp

4. **topics**: All substantive topics discussed. For each:
   - "name": Topic name (use standard/canonical naming — e.g., "Neuroplasticity" not "brain changing")
   - "relevance": "primary" (major focus), "secondary" (discussed substantially), or "mentioned" (briefly referenced)

5. **speaker_map**: Map any generic speaker labels to real names. For each:
   - "label": The label used in the transcript (e.g., "Speaker 1", "interviewer")
   - "name": The actual person's full name

Return ONLY valid JSON with these exact keys: cleaned_summary, key_claims, notable_quotes, topics, speaker_map.\
"""

NOTE_GENERATION_PROMPT = """\
You are a knowledge base note generator for an Obsidian vault called Knowledge-Web. Generate structured markdown note content using the extracted data provided.

## Existing Vault Index
{vault_index_json}

## Video Metadata
- Title: {title}
- Channel: {channel}
- Published: {date_published}
- Duration: {duration_minutes} min
- URL: {url}

## Description Mining Results
{description_mining_json}

## Transcript Processing Results
{transcript_processing_json}

## Channel Info
{channel_info_json}

## Instructions

Generate the content for all notes this video requires. Use Obsidian [[wikilink]] syntax for all cross-references.

**CRITICAL — Vault Awareness Rules:**
- Check the "Existing Vault Index" for each topic and person.
- If a topic already exists in the index, use its EXACT name in wikilinks: [[Existing Topic Name]]
- Check "topic_aliases" too — if the transcript mentions "neural plasticity" and the alias maps to "Neuroplasticity", link to [[Neuroplasticity]]
- If a person already exists in the index, link to [[Existing Person Name]]
- Only create NEW topic/person entries for ones that do NOT exist in the vault index

**Title Cleaning Rules:**
- Remove "FULL EPISODE", episode numbers (e.g., "#123"), clickbait markers
- Remove excessive capitalization — use title case
- Keep the cleaned title descriptive and searchable

Return JSON with these exact keys:

1. **video_note**: The main video note content:
   - "cleaned_title": Cleaned version of the video title (title case, no clickbait)
   - "filename": Following pattern "YYYY-MM-DD - Cleaned Title" (use the video's date_published)
   - "speakers": List of speaker names (each as string for YAML list, will be wrapped in [[]])
   - "topics": List of topic names (each as string for YAML list, will be wrapped in [[]])
   - "key_claims_count": Integer count of key claims
   - "has_scholarly_refs": Boolean — true if any book_references with type "paper" exist
   - "summary": The cleaned_summary from transcript processing (3-5 sentences)
   - "claims_markdown": Formatted markdown for the Key Claims section (numbered list, each with timestamp and speaker)
   - "quotes_markdown": Formatted markdown for the Notable Quotes section (blockquotes with attribution)
   - "chapters_markdown": Formatted markdown for the Chapters section (timestamped list)
   - "related_notes_markdown": Formatted markdown linking to related topic and person notes

2. **people**: List of person entries to create or update. For each:
   - "name": Full name
   - "is_new": true if this person does NOT exist in the vault index, false if updating
   - "role": Their role/title
   - "affiliations": List of affiliations (strings)
   - "channels": List of channels they're associated with (strings)
   - "topics": List of topics they discuss (strings)
   - "conflicts": List of conflict-of-interest notes (strings), empty list if none
   - "background": 2-3 sentence background (for new people only, null for updates)

3. **channel**: Channel note data:
   - "name": Channel name
   - "is_new": true if channel does NOT exist in vault index
   - "focus_areas": List of focus area topic names (strings)
   - "credibility_notes": One sentence about the channel's credibility/style
   - "about": 2-3 sentence channel description (for new channels only, null for updates)

4. **topics**: List of topic entries to create. ONLY include topics that are NEW (not in vault index). For each:
   - "concept": Topic name (canonical form)
   - "definition": One clear sentence defining this concept
   - "related_topics": List of related topic names (strings)
   - "aliases": List of common alternative names for this topic (lowercase)

5. **topic_updates**: List of existing topics that should have their sources_count incremented. Each is just the topic name string.

Return ONLY valid JSON with these exact keys: video_note, people, channel, topics, topic_updates.\
"""


# ═══════════════════════════════════════════
#  LLM Processor Class
# ═══════════════════════════════════════════

class LLMProcessor:
    """Runs three sequential LLM calls to process extracted video data."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8000,
        temperature: float = 0.2,
    ):
        if not api_key:
            raise RuntimeError("Anthropic API key not configured")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        """Make a single LLM call and return the text response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_llm_json(self, prompt: str) -> dict:
        """Make an LLM call and parse the response as JSON."""
        raw = self._call_llm(prompt)

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM returned invalid JSON: {e}\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            )

    # ── Step 2: Description Mining ──

    def mine_description(self, extraction_data: dict) -> dict:
        """
        LLM Call 1: Extract speakers, links, sponsors, chapters
        from the video description.

        Input: raw dict from YouTubeExtractor.extract()
        Output: structured dict with speakers, links, sponsors, etc.
        """
        metadata = extraction_data["metadata"]

        prompt = DESCRIPTION_MINING_PROMPT.format(
            title=metadata["title"],
            channel=metadata["channel_title"],
            description=metadata["description"],
        )

        print("  [LLM 1/3] Mining description...")
        result = self._call_llm_json(prompt)

        # Merge API-extracted chapters with LLM-extracted chapters
        api_chapters = metadata.get("chapters", [])
        llm_chapters = result.get("chapters", [])
        if api_chapters and not llm_chapters:
            result["chapters"] = api_chapters
        elif llm_chapters and not api_chapters:
            # Convert LLM chapters to same format
            result["chapters"] = llm_chapters
        elif api_chapters:
            # API chapters take precedence
            result["chapters"] = api_chapters

        return result

    # ── Step 3: Transcript Processing ──

    def process_transcript(
        self,
        extraction_data: dict,
        description_mining: dict,
    ) -> dict:
        """
        LLM Call 2: Clean transcript, extract key claims with timestamps,
        topic tags, map speaker labels to names.

        Input: raw extraction dict + description mining results
        Output: structured dict with summary, claims, quotes, topics, speaker_map
        """
        metadata = extraction_data["metadata"]
        transcript = extraction_data["transcript"]

        # Build known speakers context from description mining
        known_speakers = description_mining.get("speakers", [])

        # Combine API chapters + description mining chapters
        chapters = description_mining.get("chapters", metadata.get("chapters", []))

        # Truncate transcript if extremely long (>100k chars) to fit context
        timestamped_text = transcript["timestamped_text"]
        if len(timestamped_text) > 100_000:
            timestamped_text = timestamped_text[:100_000] + "\n\n[TRANSCRIPT TRUNCATED]"

        prompt = TRANSCRIPT_PROCESSING_PROMPT.format(
            title=metadata["title"],
            channel=metadata["channel_title"],
            duration_minutes=metadata["duration_minutes"],
            known_speakers_json=json.dumps(known_speakers, indent=2),
            chapters_json=json.dumps(chapters, indent=2),
            transcript=timestamped_text,
        )

        print("  [LLM 2/3] Processing transcript...")
        return self._call_llm_json(prompt)

    # ── Step 5: Vault-Aware Note Generation ──

    def generate_notes(
        self,
        extraction_data: dict,
        description_mining: dict,
        transcript_processing: dict,
        vault_index: dict,
    ) -> dict:
        """
        LLM Call 3: Generate final structured content for all notes
        this video will create or update, using vault-index.json as
        context for deduplication and linking.

        Input: all prior results + vault index
        Output: structured dict with video_note, people, channel, topics, topic_updates
        """
        metadata = extraction_data["metadata"]
        channel_info = extraction_data.get("channel", {})

        prompt = NOTE_GENERATION_PROMPT.format(
            vault_index_json=json.dumps(vault_index, indent=2),
            title=metadata["title"],
            channel=metadata["channel_title"],
            date_published=metadata["date_published"],
            duration_minutes=metadata["duration_minutes"],
            url=metadata["url"],
            description_mining_json=json.dumps(description_mining, indent=2),
            transcript_processing_json=json.dumps(transcript_processing, indent=2),
            channel_info_json=json.dumps(channel_info, indent=2),
        )

        print("  [LLM 3/3] Generating vault-aware notes...")
        return self._call_llm_json(prompt)

    # ── Full Pipeline ──

    def process(self, extraction_data: dict, vault_index: dict) -> dict:
        """
        Run all three LLM calls in sequence.

        Input:
          - extraction_data: dict from YouTubeExtractor.extract()
          - vault_index: dict loaded from _system/vault-index.json

        Output: dict with all processing results:
          {
            "description_mining": {...},
            "transcript_processing": {...},
            "note_generation": {...},
          }
        """
        # Step 1: Mine description
        description_mining = self.mine_description(extraction_data)

        # Step 2: Process transcript
        transcript_processing = self.process_transcript(
            extraction_data, description_mining
        )

        # Step 3: Generate notes
        note_generation = self.generate_notes(
            extraction_data,
            description_mining,
            transcript_processing,
            vault_index,
        )

        return {
            "description_mining": description_mining,
            "transcript_processing": transcript_processing,
            "note_generation": note_generation,
        }


# ═══════════════════════════════════════════
#  Config Loader
# ═══════════════════════════════════════════

def load_config() -> dict:
    """Load config.yaml from _system/ relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "_system", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_vault_index() -> dict:
    """Load vault-index.json from _system/ relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "_system", "vault-index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return json.load(f)
    return {
        "last_updated": None,
        "notes": {"topics": [], "people": [], "channels": [], "videos": [], "articles": []},
        "topic_aliases": {},
    }


# ═══════════════════════════════════════════
#  CLI — for standalone testing with mock data
# ═══════════════════════════════════════════

def main():
    """
    Standalone test CLI. Accepts a JSON file (output of youtube_extractor.py --json)
    and runs the three LLM calls.

    Usage:
        python youtube_extractor.py "https://youtube.com/watch?v=abc" --json > /tmp/extracted.json
        python llm_processor.py /tmp/extracted.json
        python llm_processor.py /tmp/extracted.json --step 1   # Only description mining
        python llm_processor.py /tmp/extracted.json --step 2   # Only transcript processing
        python llm_processor.py /tmp/extracted.json --step 3   # Only note generation
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Processing Pipeline — test with extracted JSON"
    )
    parser.add_argument(
        "input_json",
        help="Path to JSON file from youtube_extractor.py --json",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific step (1=description, 2=transcript, 3=notes)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (overrides config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this JSON file (default: stdout)",
    )

    args = parser.parse_args()

    # Load extraction data
    with open(args.input_json, "r") as f:
        extraction_data = json.load(f)

    # Normalize: youtube_extractor CLI wraps transcript differently
    # The extractor .extract() returns full segments, but CLI --json returns summary
    # Handle both formats
    if "transcript" in extraction_data and "segments" not in extraction_data["transcript"]:
        # CLI format — reconstruct minimal transcript dict
        extraction_data["transcript"]["segments"] = []
        if "timestamped_text" not in extraction_data["transcript"]:
            extraction_data["transcript"]["timestamped_text"] = extraction_data["transcript"].get("full_text", "")

    config = load_config()
    api_key = args.api_key or config.get("anthropic_api_key", "")

    processor = LLMProcessor(
        api_key=api_key,
        model=config.get("llm_model", "claude-sonnet-4-20250514"),
        max_tokens=config.get("llm_max_tokens", 8000),
        temperature=config.get("llm_temperature", 0.2),
    )

    vault_index = load_vault_index()

    if args.step is None:
        # Run full pipeline
        result = processor.process(extraction_data, vault_index)
    elif args.step == 1:
        result = {"description_mining": processor.mine_description(extraction_data)}
    elif args.step == 2:
        # Step 2 requires step 1 output — run step 1 first
        dm = processor.mine_description(extraction_data)
        result = {
            "description_mining": dm,
            "transcript_processing": processor.process_transcript(extraction_data, dm),
        }
    elif args.step == 3:
        # Step 3 requires steps 1+2 — run all
        dm = processor.mine_description(extraction_data)
        tp = processor.process_transcript(extraction_data, dm)
        result = {
            "description_mining": dm,
            "transcript_processing": tp,
            "note_generation": processor.generate_notes(
                extraction_data, dm, tp, vault_index
            ),
        }

    output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
