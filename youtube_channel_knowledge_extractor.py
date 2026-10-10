#!/usr/bin/env python3
"""
YouTube Channel Knowledge Extractor (monolithic single-file build)

This file is every module of the ytkx package merged into one file, for
quick hand-off to another machine or another assistant (such as Claude
Code). It behaves exactly like the multi-file project: run it the same way.

Usage:
    python3 youtube_channel_knowledge_extractor.py \
        --channel https://www.youtube.com/@ChannelName

Install first:
    pip install -r requirements.txt
    (see the requirements list near the end of this file's companion
    README, or: yt-dlp youtube-transcript-api anthropic
    sentence-transformers scikit-learn numpy)

Optional: set ANTHROPIC_API_KEY so the AI steps (5 and 6) run at full
quality; without it the tool falls back to simple offline methods.

Where things are saved:
    work-in-progress/<channel>_<timestamp>/   every intermediate file one
        run produces (listing, transcripts, outline, manuscript drafts).
        Each run gets its own fresh, timestamped folder, so earlier test
        runs are kept, not overwritten. Pass --work-dir to reuse a specific
        earlier run's folder instead (for example, to resume with
        --from-step/--to-step).
    output/             one finished-book copy per run, named
                        @channel_YYYY-MM-DD_HH-MM-SS.txt
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
import argparse
import json
import numpy as np
import os
import re
import shutil
import time
import yt_dlp

# ============================================================================
# From ytkx/config.py
# ============================================================================
@dataclass
class Settings:
    channel_url: str
    work_dir: Path = Path("work-in-progress")
    output_dir: Path = Path("output")
    languages: tuple = ("en",)
    model: str = os.environ.get("YTKX_MODEL", "claude-sonnet-5")
    use_llm: bool = True
    similarity_threshold: float = 0.88
    min_sentence_words: int = 4
    delay_seconds: float = 1.0
    max_videos: int = 0  # 0 means no limit

    @property
    def listing_file(self) -> Path:
        return self.work_dir / "01_listing.json"

    @property
    def urls_file(self) -> Path:
        return self.work_dir / "01_all_urls.txt"

    @property
    def organized_file(self) -> Path:
        return self.work_dir / "02_organized.json"

    @property
    def raw_dir(self) -> Path:
        return self.work_dir / "transcripts_raw"

    @property
    def clean_dir(self) -> Path:
        # The "sister directory" of transcripts_raw.
        return self.work_dir / "transcripts_clean"

    @property
    def manifest_file(self) -> Path:
        return self.work_dir / "04_manifest.json"

    @property
    def outline_file(self) -> Path:
        return self.work_dir / "06_outline.json"

    @property
    def sentences_file(self) -> Path:
        return self.work_dir / "07_unique_sentences.json"

    @property
    def removed_file(self) -> Path:
        return self.work_dir / "07_removed_log.json"

    @property
    def manuscript_dir(self) -> Path:
        return self.work_dir / "manuscript"

# ============================================================================
# From ytkx/util.py
# ============================================================================
def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "", flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_len] or "untitled"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json(text: str):
    """Pull the first JSON object out of a model reply. Returns None on failure."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def next_version_path(directory: Path, stem: str, ext: str) -> Path:
    """Return directory/stem_vN.ext where N is one higher than any existing version."""
    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+){re.escape(ext)}$")
    numbers = []
    for p in directory.iterdir():
        m = pattern.match(p.name)
        if m:
            numbers.append(int(m.group(1)))
    return directory / f"{stem}_v{max(numbers, default=0) + 1}{ext}"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def write_transcript_file(path: Path, title: str, url: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Title: {title}\nURL: {url}\n\n{body}\n", encoding="utf-8")


def read_transcript_body(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else raw.strip()

# ============================================================================
# From ytkx/llm.py
# ============================================================================
class LLM:
    """Thin wrapper around the Anthropic API."""

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def ask(self, system: str, user: str, max_tokens: int = 4096) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()


def get_llm(settings: Settings):
    """Return an LLM helper, or None when running without an API key or with --no-llm."""
    if not settings.use_llm:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found. Using simple offline fallbacks.")
        return None
    return LLM(settings.model)

# ============================================================================
# From ytkx/embed.py
# ============================================================================
_model = None


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Turn sentences into number lists (embeddings) so meaning can be compared."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return np.asarray(
        _model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    )

# ============================================================================
# From ytkx/step1_list.py
# ============================================================================
def normalize_channel_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return re.sub(r"/(videos|playlists|streams|shorts|featured|home)$", "", url)


def extract_playlist_id(url: str) -> str | None:
    """Return the playlist id if the URL names one (?list=... or &list=...), else None."""
    match = re.search(r"[?&]list=([A-Za-z0-9_-]+)", url)
    return match.group(1) if match else None


def channel_handle(channel_url: str) -> str:
    """Turn a channel or playlist URL into a short label for file names.

    Examples: https://www.youtube.com/@SeanDollwet -> @SeanDollwet
              .../watch?v=...&list=PLGSTqsa4O_EQ -> playlist_PLGSTqsa4O_EQ
    """
    playlist_id = extract_playlist_id(channel_url)
    if playlist_id:
        return f"playlist_{playlist_id}"
    last = normalize_channel_url(channel_url).rstrip("/").rsplit("/", 1)[-1]
    if not last.startswith("@"):
        last = "@" + last
    return re.sub(r"[^\w@-]", "_", last)


def _extract(url: str) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",  # list entries without downloading each video page
        "skip_download": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _video_record(entry: dict) -> dict:
    vid = entry["id"]
    return {
        "id": vid,
        "title": entry.get("title") or vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "duration": entry.get("duration"),
        "upload_date": entry.get("upload_date"),
    }


def _videos_from(info: dict) -> list[dict]:
    videos = []
    for entry in info.get("entries") or []:
        # YouTube video ids are always 11 characters. Playlist ids are longer.
        if entry and entry.get("id") and len(entry["id"]) == 11:
            videos.append(_video_record(entry))
    return videos


def list_channel(settings: Settings) -> dict:
    base = normalize_channel_url(settings.channel_url)

    print(f"Listing uploads for {base} ...")
    uploads = _extract(base + "/videos")
    videos = _videos_from(uploads)
    if settings.max_videos:
        videos = videos[: settings.max_videos]

    print("Looking for playlists ...")
    playlists = []
    for entry in _extract(base + "/playlists").get("entries") or []:
        if not entry or not entry.get("id"):
            continue
        url = entry.get("url") or f"https://www.youtube.com/playlist?list={entry['id']}"
        members = _videos_from(_extract(url))
        if settings.max_videos:
            members = members[: settings.max_videos]
        if members:
            playlists.append(
                {"id": entry["id"], "title": entry.get("title") or entry["id"], "url": url, "videos": members}
            )

    listing = {
        "channel_title": uploads.get("channel") or uploads.get("title") or "YouTube Channel",
        "channel_url": base,
        "videos": videos,
        "playlists": playlists,
    }
    write_json(settings.listing_file, listing)
    all_urls = [v["url"] for v in videos] + [v["url"] for p in playlists for v in p["videos"]]
    settings.urls_file.write_text("\n".join(all_urls) + "\n", encoding="utf-8")
    print(f"Found {len(videos)} unplaylisted videos and {len(playlists)} playlists.")
    return listing


def list_playlist(settings: Settings) -> dict:
    """List a single playlist (or a watch URL that names one via ?list=...) directly."""
    playlist_id = extract_playlist_id(settings.channel_url)
    url = f"https://www.youtube.com/playlist?list={playlist_id}"

    print(f"Listing playlist {url} ...")
    info = _extract(url)
    videos = _videos_from(info)
    if settings.max_videos:
        videos = videos[: settings.max_videos]
    title = info.get("title") or f"Playlist {playlist_id}"

    listing = {
        "channel_title": title,
        "channel_url": url,
        "videos": [],
        "playlists": [{"id": playlist_id, "title": title, "url": url, "videos": videos}],
    }
    write_json(settings.listing_file, listing)
    settings.urls_file.write_text("\n".join(v["url"] for v in videos) + "\n", encoding="utf-8")
    print(f"Found {len(videos)} videos in the playlist.")
    return listing

# ============================================================================
# From ytkx/step2_3_organize.py
# ============================================================================
def cluster_by_title(videos: list[dict], target_size: int = 10) -> list[dict]:
    """Group videos by similar words in their titles (TF-IDF plus KMeans clustering)."""
    if len(videos) < 6:
        return [{"name": "General", "kind": "topic", "videos": videos}]

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    titles = [v["title"] for v in videos]
    k = max(2, min(30, round(len(videos) / target_size), len(videos)))
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(titles)
    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matrix)
    terms = vectorizer.get_feature_names_out()

    groups = []
    for cluster_id in range(k):
        members = [v for v, label in zip(videos, model.labels_) if label == cluster_id]
        if not members:
            continue
        top = model.cluster_centers_[cluster_id].argsort()[::-1][:3]
        name = ", ".join(terms[i] for i in top).title()
        groups.append({"name": name, "kind": "topic", "videos": members})
    return groups


def organize(listing: dict, settings: Settings) -> dict:
    known = {v["id"]: v for v in listing["videos"]}
    assigned: set[str] = set()
    groups: list[dict] = []

    # Step 2: playlists. A video in several playlists is kept only in the first one.
    for playlist in listing["playlists"]:
        members = []
        for video in playlist["videos"]:
            if video["id"] in assigned:
                continue
            assigned.add(video["id"])
            members.append(known.get(video["id"], video))
        if members:
            groups.append({"name": playlist["title"], "kind": "playlist", "videos": members})

    leftovers = [v for v in listing["videos"] if v["id"] not in assigned]
    if leftovers:
        if listing["playlists"]:
            groups.append({"name": "Videos Not In Any Playlist", "kind": "unlisted", "videos": leftovers})
        else:
            # Step 3: no playlists at all, so organize logically by topic.
            groups.extend(cluster_by_title(leftovers))

    organized = {"channel_title": listing["channel_title"], "groups": groups}
    write_json(settings.organized_file, organized)
    for g in groups:
        print(f"  {g['name']} ({g['kind']}): {len(g['videos'])} videos")
    return organized

# ============================================================================
# From ytkx/step4_transcripts.py
# ============================================================================
def tidy_caption_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)  # remove tags such as [Music] and [Applause]
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _ytdlp_caption_track(video_id: str, languages: list[str]) -> tuple[str, str] | None:
    """Ask yt-dlp (not youtube-transcript-api) for a json3 caption track URL.

    Tries real captions before auto-generated ones, and the requested languages
    in order. Returns (language_code, url), or None if nothing usable was found.
    """
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False) or {}

    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        for lang in languages:
            for code, formats in tracks.items():
                if code == lang or code.startswith(lang + "-"):
                    for fmt in formats:
                        if fmt.get("ext") == "json3" and fmt.get("url"):
                            return code, fmt["url"]
    return None


def _fetch_json3_captions(url: str) -> str:
    """Download a json3 caption track and join its text segments into one string."""
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    pieces = [
        seg.get("utf8", "")
        for event in data.get("events", [])
        for seg in (event.get("segs") or [])
    ]
    return "".join(pieces)


def fetch_transcript_text(api: YouTubeTranscriptApi, video_id: str, languages: list[str]) -> str:
    """Fetch a transcript via youtube-transcript-api, falling back to yt-dlp's own
    caption tracks (a different request path) if that is blocked or fails."""
    try:
        fetched = api.fetch(video_id, languages=languages)
        return tidy_caption_text(" ".join(snippet.text for snippet in fetched))
    except Exception as primary_exc:
        track = _ytdlp_caption_track(video_id, languages)
        if not track:
            raise primary_exc
        _, url = track
        return tidy_caption_text(_fetch_json3_captions(url))


MAX_CONSECUTIVE_FAILURES = 3  # stop hammering a rate-limited IP instead of failing through the whole list
MAX_BACKOFF_SECONDS = 60.0


def download_transcripts(settings: Settings, organized: dict) -> dict:
    api = YouTubeTranscriptApi()
    manifest = read_json(settings.manifest_file) if settings.manifest_file.exists() else {}

    consecutive_failures = 0
    stopped_early = False

    for index, group in enumerate(organized["groups"], start=1):
        folder = f"{index:02d}_{slugify(group['name'])}"
        for video in group["videos"]:
            relative = f"{folder}/{video['id']}_{slugify(video['title'])}.txt"
            path = settings.raw_dir / relative
            entry = {
                "group": group["name"],
                "title": video["title"],
                "url": video["url"],
                "file": relative,
                "status": "ok",
            }
            if path.exists():  # resume support: skip what is already downloaded
                manifest[video["id"]] = entry
                consecutive_failures = 0
                continue

            if stopped_early:
                entry["status"] = "skipped"
                entry["error"] = "stopped_early_after_repeated_failures"
                manifest[video["id"]] = entry
                continue

            try:
                body = fetch_transcript_text(api, video["id"], list(settings.languages))
                write_transcript_file(path, video["title"], video["url"], body)
                print(f"  saved {relative}")
                consecutive_failures = 0
                manifest[video["id"]] = entry
                write_json(settings.manifest_file, manifest)
                time.sleep(settings.delay_seconds)
                continue
            except Exception as exc:  # captions off, private video, blocked IP, and so on
                entry["status"] = "failed"
                entry["error"] = type(exc).__name__
                consecutive_failures += 1
                print(f"  FAILED {video['id']}: {type(exc).__name__} ({consecutive_failures} failure(s) in a row)")
                manifest[video["id"]] = entry
                write_json(settings.manifest_file, manifest)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                stopped_early = True
                print(
                    f"  Stopping transcript downloads after {consecutive_failures} failures in a row - "
                    "this usually means YouTube is rate-limiting this IP right now, not that every "
                    "remaining video lacks captions. Remaining videos are marked 'skipped'. Wait a "
                    "while (try tens of minutes to a few hours) before re-running this step."
                )
                continue

            backoff = min(settings.delay_seconds * (2 ** consecutive_failures), MAX_BACKOFF_SECONDS)
            print(f"  Backing off for {backoff:.0f}s before the next request ...")
            time.sleep(backoff)

    write_json(settings.manifest_file, manifest)
    ok = sum(1 for m in manifest.values() if m["status"] == "ok")
    failed = sum(1 for m in manifest.values() if m["status"] == "failed")
    skipped = sum(1 for m in manifest.values() if m["status"] == "skipped")
    print(f"Transcripts saved: {ok} of {len(manifest)} ({failed} failed, {skipped} skipped).")
    return manifest

# ============================================================================
# From ytkx/step5_clean.py
# ============================================================================
SYSTEM = (
    "You are a careful copy editor. You receive raw, unpunctuated speech-to-text captions. "
    "Rewrite them as complete, grammatically correct sentences with proper capitalization "
    "and punctuation. Fix obvious mishearings only when the intended word is clear. "
    "Do not summarize, do not add ideas, and do not drop ideas. Remove filler words such as "
    "um and uh. Output only the rewritten text as plain paragraphs, with no headings and no commentary."
)


def chunk_words(text: str, size: int = 1200) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def basic_clean(text: str) -> str:
    """Crude offline fallback: cut into roughly 20 word sentences, capitalize, add periods."""
    words = text.split()
    sentences = []
    for i in range(0, len(words), 20):
        piece = " ".join(words[i : i + 20]).strip()
        if piece:
            piece = piece[0].upper() + piece[1:]
            piece = re.sub(r"\bi\b", "I", piece)
            sentences.append(piece.rstrip(".,;:") + ".")
    return " ".join(sentences)


def clean_all(settings: Settings, llm) -> None:
    manifest = read_json(settings.manifest_file)
    for video_id, entry in manifest.items():
        if entry["status"] != "ok":
            continue
        target = settings.clean_dir / entry["file"]
        if target.exists():
            continue
        raw = read_transcript_body(settings.raw_dir / entry["file"])
        if llm:
            cleaned = "\n\n".join(llm.ask(SYSTEM, chunk, max_tokens=4096) for chunk in chunk_words(raw))
        else:
            cleaned = basic_clean(raw)
        write_transcript_file(target, entry["title"], entry["url"], cleaned)
        print(f"  cleaned {entry['file']}")

# ============================================================================
# From ytkx/step6_outline.py
# ============================================================================
TOPIC_SYSTEM = "You extract topic headings from transcripts. Reply with JSON only."
MERGE_SYSTEM = "You design clean book outlines. Reply with JSON only."


def iter_leaves(group: dict):
    """The leaves are the lowest headings that hold text: subsections, or a section with none."""
    for section in group["sections"]:
        subsections = section.get("subsections") or []
        if subsections:
            for sub in subsections:
                yield sub
        else:
            yield section


def _video_topics(llm, title: str, text: str) -> list[str]:
    prompt = (
        f"Video title: {title}\n\nTranscript:\n{text[:12000]}\n\n"
        'List 4 to 10 short topic headings covering the main ideas. Reply as JSON: {"topics": ["..."]}'
    )
    data = extract_json(llm.ask(TOPIC_SYSTEM, prompt, max_tokens=800)) or {}
    return [t for t in data.get("topics", []) if isinstance(t, str)]


def _merge_topics(llm, group_name: str, topics: list[str]):
    unique = list(dict.fromkeys(t.strip() for t in topics if t.strip()))[:300]
    listing = "\n".join(f"- {t}" for t in unique)
    prompt = (
        f"Group name: {group_name}\n\nTopic headings collected from the videos:\n{listing}\n\n"
        "Merge related and duplicate headings into a clean outline with 2 to 12 sections, "
        "each with 1 to 6 subsections. Order them so the material reads logically from "
        'foundations to advanced ideas. Reply as JSON: {"sections": [{"title": "...", "subsections": ["..."]}]}'
    )
    data = extract_json(llm.ask(MERGE_SYSTEM, prompt, max_tokens=3000)) or {}
    sections = []
    for sec in data.get("sections", []):
        if not isinstance(sec, dict) or not sec.get("title"):
            continue
        subs = [{"title": s} for s in sec.get("subsections", []) if isinstance(s, str)]
        sections.append({"title": sec["title"], "subsections": subs})
    return sections


def _fallback_sections(videos: list[tuple[str, str]]) -> list[dict]:
    # One section per video. Sentences will be placed by which video they came from.
    return [{"title": title, "video_id": vid, "subsections": []} for vid, title in videos]


def build_outline(settings: Settings, organized: dict, llm) -> dict:
    manifest = read_json(settings.manifest_file)
    outline = {"channel_title": organized["channel_title"], "groups": []}

    for group in organized["groups"]:
        videos = [
            (v["id"], v["title"])
            for v in group["videos"]
            if manifest.get(v["id"], {}).get("status") == "ok"
        ]
        if not videos:
            continue

        sections = []
        if llm:
            topics = []
            for vid, title in videos:
                body = read_transcript_body(settings.clean_dir / manifest[vid]["file"])
                topics.extend(_video_topics(llm, title, body))
            sections = _merge_topics(llm, group["name"], topics)
        if not sections:
            sections = _fallback_sections(videos)
        outline["groups"].append({"name": group["name"], "sections": sections})
        print(f"  outline for {group['name']}: {len(sections)} sections")

    write_json(settings.outline_file, outline)
    return outline

# ============================================================================
# From ytkx/step7_dedupe.py
# ============================================================================
def load_sentences(settings: Settings) -> list[dict]:
    manifest = read_json(settings.manifest_file)
    sentences, order = [], 0
    for video_id, entry in manifest.items():
        if entry["status"] != "ok":
            continue
        body = read_transcript_body(settings.clean_dir / entry["file"])
        for text in split_sentences(body):
            sentences.append(
                {"text": text, "group": entry["group"], "video_id": video_id, "title": entry["title"], "order": order}
            )
            order += 1
    return sentences


def remove_duplicates(sentences: list[dict], settings: Settings):
    removed, seen_exact, candidates = [], {}, []

    for s in sentences:
        key = re.sub(r"[^a-z0-9 ]", "", s["text"].lower()).strip()
        if len(key.split()) < settings.min_sentence_words:
            removed.append({**s, "reason": "too short to stand alone"})
        elif key in seen_exact:
            removed.append({**s, "reason": "exact duplicate", "matched": seen_exact[key]})
        else:
            seen_exact[key] = s["text"]
            candidates.append(s)

    if not candidates:
        return [], removed

    vectors = embed([c["text"] for c in candidates])
    kept_vectors = np.zeros_like(vectors)
    kept_index: list[int] = []

    for i, candidate in enumerate(candidates):
        if kept_index:
            similarity = kept_vectors[: len(kept_index)] @ vectors[i]
            best = int(np.argmax(similarity))
            if similarity[best] >= settings.similarity_threshold:
                removed.append(
                    {
                        **candidate,
                        "reason": "same meaning as an earlier sentence",
                        "similarity": round(float(similarity[best]), 3),
                        "matched": candidates[kept_index[best]]["text"],
                    }
                )
                continue
        kept_vectors[len(kept_index)] = vectors[i]
        kept_index.append(i)

    return [candidates[i] for i in kept_index], removed


def dedupe_all(settings: Settings) -> list[dict]:
    sentences = load_sentences(settings)
    kept, removed = remove_duplicates(sentences, settings)
    write_json(settings.sentences_file, kept)
    write_json(settings.removed_file, removed)
    print(f"  kept {len(kept)} sentences, removed {len(removed)} of {len(sentences)}")
    return kept

# ============================================================================
# From ytkx/step8_manuscript.py
# ============================================================================
SENTENCES_PER_PARAGRAPH = 5


def assign_sentences(outline: dict, sentences: list[dict]) -> None:
    by_group = defaultdict(list)
    for s in sentences:
        by_group[s["group"]].append(s)

    for group in outline["groups"]:
        leaves = list(iter_leaves(group))
        for leaf in leaves:
            leaf["sentences"] = []
        group_sentences = by_group.get(group["name"], [])
        if not leaves or not group_sentences:
            continue

        if all("video_id" in leaf for leaf in leaves):
            # Fallback outline: place sentences by the video they came from.
            by_video = {leaf["video_id"]: leaf for leaf in leaves}
            for s in group_sentences:
                if s["video_id"] in by_video:
                    by_video[s["video_id"]]["sentences"].append(s)
            continue

        # Normal outline: place each sentence under the heading closest in meaning.
        heading_vectors = embed([leaf["title"] for leaf in leaves])
        sentence_vectors = embed([s["text"] for s in group_sentences])
        best = np.argmax(sentence_vectors @ heading_vectors.T, axis=1)
        for s, leaf_index in zip(group_sentences, best):
            leaves[int(leaf_index)]["sentences"].append(s)


def _has_text(section: dict) -> bool:
    return any(leaf.get("sentences") for leaf in iter_leaves({"sections": [section]}))


def _paragraphs(sentences: list[dict]) -> list[str]:
    ordered = [s["text"] for s in sorted(sentences, key=lambda s: s["order"])]
    return [
        " ".join(ordered[i : i + SENTENCES_PER_PARAGRAPH])
        for i in range(0, len(ordered), SENTENCES_PER_PARAGRAPH)
    ]


def write_manuscript(settings: Settings, outline: dict) -> Path:
    toc: list[str] = []
    body: list[str] = []

    group_number = 0
    for group in outline["groups"]:
        sections = [s for s in group["sections"] if _has_text(s)]
        if not sections:
            continue
        group_number += 1
        toc.append(f"{group_number}. {group['name']}")
        body.append(f"{group_number}. {group['name']}\n")

        for si, section in enumerate(sections, start=1):
            toc.append(f"{group_number}.{si} {section['title']}")
            body.append(f"{group_number}.{si} {section['title']}\n")
            subsections = [s for s in (section.get("subsections") or []) if s.get("sentences")]
            if not subsections:
                body.extend(p + "\n" for p in _paragraphs(section.get("sentences", [])))
                continue
            for ki, sub in enumerate(subsections, start=1):
                toc.append(f"{group_number}.{si}.{ki} {sub['title']}")
                body.append(f"{group_number}.{si}.{ki} {sub['title']}\n")
                body.extend(p + "\n" for p in _paragraphs(sub["sentences"]))

    title = outline["channel_title"]
    text = "\n".join([title, "Master Manuscript", "", "Table of Contents", "", *toc, "", *body])
    path = next_version_path(settings.manuscript_dir, "master_manuscript", ".txt")
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Manuscript written to {path}")
    return path


def save_output_copy(settings: Settings, manuscript_path: Path) -> Path:
    """Copy the finished book into output/, named @channel_YYYY-MM-DD_HH-MM-SS.txt."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = settings.output_dir / f"{channel_handle(settings.channel_url)}_{stamp}.txt"
    shutil.copyfile(manuscript_path, out_path)
    print(f"Book copy saved to {out_path}")
    return out_path

# ============================================================================
# From ytkx/cli.py
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ytkx", description="YouTube Channel Knowledge Extractor: channel to master manuscript."
    )
    p.add_argument(
        "--channel",
        required=True,
        help=(
            "Channel URL (https://www.youtube.com/@name) or a playlist/watch URL "
            "containing list=... to process just that one playlist"
        ),
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Folder for intermediate files. Default: a fresh, timestamped folder under "
            "work-in-progress/ is created for every run, so earlier test runs are kept. "
            "Pass an existing run's folder here to resume it with --from-step/--to-step."
        ),
    )
    p.add_argument(
        "--output-dir", default="output", help="Folder where each run's finished book copy is saved (default: output)"
    )
    p.add_argument("--languages", default="en", help="Comma separated caption languages (default: en)")
    p.add_argument("--from-step", type=int, default=1, help="First step to run, 1 to 8 (default: 1)")
    p.add_argument("--to-step", type=int, default=8, help="Last step to run, 1 to 8 (default: 8)")
    p.add_argument("--no-llm", action="store_true", help="Skip the AI model and use simple offline fallbacks")
    p.add_argument("--threshold", type=float, default=0.88, help="Duplicate similarity from 0 to 1 (default: 0.88)")
    p.add_argument(
        "--delay", type=float, default=2.0, help="Seconds to wait between transcript requests (default: 2.0)"
    )
    p.add_argument("--max-videos", type=int, default=0, help="Limit videos for a quick test (0 means all)")
    return p.parse_args(argv)


def default_work_dir(channel_url: str) -> Path:
    """A fresh, timestamped folder for one run, so earlier test runs are never overwritten."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("work-in-progress") / f"{channel_handle(channel_url)}_{stamp}"


def main(argv=None) -> int:
    a = parse_args(argv)
    work_dir = Path(a.work_dir) if a.work_dir else default_work_dir(a.channel)
    settings = Settings(
        channel_url=a.channel,
        work_dir=work_dir,
        output_dir=Path(a.output_dir),
        languages=tuple(x.strip() for x in a.languages.split(",") if x.strip()),
        use_llm=not a.no_llm,
        similarity_threshold=a.threshold,
        delay_seconds=a.delay,
        max_videos=a.max_videos,
    )
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working files for this run: {settings.work_dir}")
    llm = get_llm(settings) if a.to_step >= 5 and a.from_step <= 6 else None
    run = lambda n: a.from_step <= n <= a.to_step  # noqa: E731

    if run(1):
        if extract_playlist_id(settings.channel_url):
            print("Step 1: listing the playlist")
            listing = list_playlist(settings)
        else:
            print("Step 1: listing the channel")
            listing = list_channel(settings)
    elif settings.listing_file.exists():
        listing = read_json(settings.listing_file)

    if run(2) or run(3):
        print("Steps 2 and 3: organizing videos")
        organized = organize(listing, settings)
    elif settings.organized_file.exists():
        organized = read_json(settings.organized_file)

    if run(4):
        print("Step 4: downloading transcripts")
        download_transcripts(settings, organized)

    if run(5):
        print("Step 5: cleaning transcripts into grammatical sentences")
        clean_all(settings, llm)

    if run(6):
        print("Step 6: building the master outline")
        outline = build_outline(settings, organized, llm)
    elif settings.outline_file.exists():
        outline = read_json(settings.outline_file)

    if run(7):
        print("Step 7: removing duplicate ideas")
        sentences = dedupe_all(settings)
    elif settings.sentences_file.exists():
        sentences = read_json(settings.sentences_file)

    if run(8):
        print("Step 8: writing the master manuscript")
        assign_sentences(outline, sentences)
        manuscript_path = write_manuscript(settings, outline)
        save_output_copy(settings, manuscript_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
