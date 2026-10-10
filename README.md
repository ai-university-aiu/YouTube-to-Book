# YouTube-to-Book

YouTube Channel Knowledge Extractor: turn a whole YouTube channel into one organized master manuscript.

## What it does

Given a channel URL, the tool works through 8 steps:

1. List every video and playlist on the channel.
2. Group videos by playlist.
3. For videos in no playlist, group them by topic (TF-IDF + KMeans clustering on titles).
4. Download each video's captions as a transcript.
5. Clean each transcript into grammatical sentences (uses the Anthropic API if `ANTHROPIC_API_KEY` is set, otherwise a simple offline fallback).
6. Build a master outline of sections and subsections (same AI-or-fallback choice as step 5).
7. Remove duplicate and near-duplicate sentences using sentence embeddings.
8. Assemble everything into a single master manuscript text file, with a table of contents.

Each step's output is saved to disk, so a run can be resumed or re-run from any step with `--from-step` / `--to-step`.

## Where files are saved

- `work-in-progress/<channel>_<timestamp>/` - every intermediate file one run produces: the video listing, raw and cleaned transcripts, the outline, and versioned manuscript drafts (`master_manuscript_v1.txt`, `v2.txt`, and so on). Every run gets its own fresh, timestamped folder here automatically, so earlier test runs are kept side by side, never overwritten. The folder actually used is printed at the start of each run.
- `output/` - one finished-book copy per completed run, named `@channel_YYYY-MM-DD_HH-MM-SS.txt`, for example `@SeanDollwet_2026-10-10_15-42-07.txt`. This folder is the one to look in for the finished result.

To resume a specific earlier run instead of starting fresh (for example with `--from-step`/`--to-step`), pass that run's exact folder back in with `--work-dir`:

```bash
python3 youtube_channel_knowledge_extractor.py --channel https://www.youtube.com/@ChannelName \
    --work-dir "work-in-progress/@ChannelName_2026-10-10_15-54-46" --from-step 5 --to-step 8
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional, for full-quality AI cleaning and outlining in steps 5 and 6:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Without a key, the tool automatically falls back to simple offline methods and still produces a complete manuscript.

## Usage

Quick test on a small slice of a channel:

```bash
python3 youtube_channel_knowledge_extractor.py --channel https://www.youtube.com/@ChannelName --max-videos 5
```

Full run:

```bash
python3 youtube_channel_knowledge_extractor.py --channel https://www.youtube.com/@ChannelName
```

A single playlist can be processed on its own, instead of a whole channel, by passing any URL that contains `list=...` (a playlist page, or a video's watch URL that names a playlist):

```bash
python3 youtube_channel_knowledge_extractor.py --channel "https://www.youtube.com/watch?v=VIDEO_ID&list=PLAYLIST_ID" --max-videos 5
```

### Options

- `--work-dir FOLDER` - where intermediate files are saved (default: a new timestamped folder under `work-in-progress/` for every run)
- `--output-dir FOLDER` - where each run's finished book copy is saved (default: `output`)
- `--languages en,es` - caption languages to try, in order
- `--threshold 0.88` - how alike two sentences must be to count as the same idea; raise toward `0.95` to remove less, lower toward `0.80` to remove more
- `--delay 1.0` - seconds to wait between transcript requests; raise it if YouTube starts refusing
- `--from-step` / `--to-step` - rerun only part of the pipeline, numbered 1 through 8
- `--no-llm` - run without the AI model, using simple offline fallbacks

## A known limitation: cloud IP blocking

Step 4, downloading transcripts, calls YouTube directly for each video's captions. YouTube blocks most requests coming from cloud-provider IP addresses (Amazon Web Services, Google Cloud Platform, Microsoft Azure, and similar hosted machines), regardless of this tool. If every video fails with `IpBlocked`, this is almost always the cause, not a bug in the code. Run the tool from a normal home or office internet connection instead, or see the "Working around IP bans" section of the youtube-transcript-api project for other options.

## Responsible use

Transcripts and the manuscript built from them are based on other people's creative work. Use this tool on your own channel or with the creator's permission, and follow YouTube's Terms of Service. Review the manuscript before publishing anything under your name, since the AI grammar step rewrites text.

## Attribution

This tool is a fresh, single-file build of the YouTube Channel Knowledge Extractor project.
