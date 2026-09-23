#!/usr/bin/env python3
"""Find matched interactive vs single-shot LLM conversations.

Default source: allenai/WildChat (streaming).
Optional source: lmsys/lmsys-chat-1m (requires accepting its HF terms).

Each call returns up to 10 NEW pairs and appends selected prompt hashes and
records to JSONL files so subsequent calls do not repeat them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from utils import clean_text


from adapters import ADAPTERS

def get_adapter(source):
    try:
        return ADAPTERS[source]()
    except KeyError:
        raise ValueError(
            f"No adapter defined for {source}"
        )

logging.basicConfig(
    filename="Data/diagnostics.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

QUESTION_WORDS = re.compile(
    r"(?i)\b(?:what|which|who|whom|whose|when|where|why|how|could you|"
    r"can you|would you|do you|are you|is it|please provide|please specify|"
    r"please clarify|can you clarify|could you clarify|tell me more)\b"
)
INFO_REQUESTS = re.compile(
    r"(?i)\b(?:if you (?:can |could |would )?(?:provide|share|tell)|"
    r"if you provide me with|to (?:better )?(?:help|answer|assist)|"
    r"need (?:more|additional|some) (?:information|details|context)|"
    r"more information|additional details|further context|"
    r"it would help (?:to know|if)|depends on|could you elaborate|"
    r"what do you mean|do you mean)\b"
)
GENERIC_OFFER = re.compile(
    r"(?i)^(?:let me know if you have any questions|"
    r"is there anything else i can help you with)[.!? ]*$"
)

DOMAIN_PATTERNS: Dict[str, re.Pattern] = {
    "medicine": re.compile(r"(?i)\b(?:patient|diagnos|symptom|treatment|medicine|clinical|doctor|blood test|dose|disease|health)\b"),
    "law": re.compile(r"(?i)\b(?:law|legal|court|contract|statute|jurisdiction|attorney|lawyer|lawsuit|regulation)\b"),
    "education": re.compile(r"(?i)\b(?:student|teacher|lesson|curriculum|homework|exam|course|grade|tutor|learning objective)\b"),
    #"programming": re.compile(r"(?i)\b(?:python|javascript|java|code|debug|compiler|api|database|function|exception|software)\b"),
    "research": re.compile(r"(?i)\b(?:study|research|hypothesis|dataset|experiment|statistical|model|methodology|paper)\b"),
    "finance": re.compile(r"(?i)\b(?:investment|portfolio|tax|mortgage|budget|finance|stock|interest rate|loan)\b"),
    "travel": re.compile(r"(?i)\b(?:travel|trip|hotel|flight|itinerary|visa|destination|airport)\b"),
    "writing": re.compile(r"(?i)\b(?:write|rewrite|email|essay|letter|tone|audience|proofread|paragraph)\b"),
}

@dataclass
class Candidate:
    source: str
    conversation_id: str
    model: str
    language: str
    domain: str
    initial_prompt: str
    first_response: str
    remaining_messages: List[Dict[str, str]]
    full_messages: List[Dict[str, str]]

@dataclass
class Pair:
    interactive: Candidate
    single_shot: Candidate
    similarity: float

EMBEDDING_MODEL = None

def get_embedding_model():
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return EMBEDDING_MODEL




def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def alternates_user_assistant(messages: Sequence[Dict[str, str]]) -> bool:
    if len(messages) < 2 or messages[0]["role"] != "user":
        return False
    return all(m["role"] != messages[i - 1]["role"] for i, m in enumerate(messages[1:], 1))


def is_safe_row(
    row: Dict[str, Any],
    messages: Sequence[Dict[str, str]],
) -> bool:
    """
    Return True only when the conversation is not marked toxic,
    redacted, or moderation-flagged.

    Comparisons use `is True` so missing values, empty containers,
    and other non-Boolean values are not accidentally interpreted
    as positive safety flags.
    """

    # Conversation-level flags
    if row.get("toxic") is True:
        return False

    if row.get("redacted") is True:
        return False

    # Message-level flags from the original dataset row
    raw_messages = (
        row.get("conversation")
        or row.get("conversations")
        or row.get("messages")
        or []
    )

    for message in raw_messages:
        if not isinstance(message, dict):
            continue

        if message.get("toxic") is True:
            return False

        if message.get("redacted") is True:
            return False

    # OpenAI moderation metadata
    moderation_results = (
        row.get("openai_moderation")
        or []
    )

    for result in moderation_results:
        if not isinstance(result, dict):
            continue

        if result.get("flagged") is True:
            return False

        categories = result.get("categories") or {}

        if isinstance(categories, dict):
            if any(
                value is True
                for value in categories.values()
            ):
                return False

    return True


def is_clarification(
    response: str
) -> bool:

    text = clean_text(
        response
    )

    if not text:
        return False

    if (
        GENERIC_OFFER.fullmatch(
            text
        )
    ):
        return False

    if not text.endswith("?"):
        return False

    return bool(
        QUESTION_WORDS.search(
            text
        )
        or
        INFO_REQUESTS.search(
            text
        )
    )


def infer_domain(text: str) -> str:
    scores = {name: len(pattern.findall(text)) for name, pattern in DOMAIN_PATTERNS.items()}
    domain, score = max(scores.items(), key=lambda x: x[1])
    return domain if score > 0 else "general"


def row_id(row: Dict[str, Any]) -> str:
    value = row.get("conversation_id") or row.get("conversation_hash") or row.get("id")
    if value:
        return str(value)
    return text_hash(json.dumps(row, sort_keys=True, default=str))[:24]


def row_to_candidate(
    row: Dict[str, Any],
    source: str,
    adapter,
) -> Tuple[Optional[Candidate], str]:

    messages = adapter.normalize_messages(row)

    if not messages:
        return None, "empty_messages"

    if not alternates_user_assistant(messages):
        return None, "failed_alternation"

    if not is_safe_row(row, messages):
        return None, "failed_safety"

    prompt = messages[0]["content"]

    if len(prompt.split()) < 3:
        return None, "short_prompt"

    context = " ".join(
        message["content"]
        for message in messages[:4]
    )

    candidate = Candidate(
        source=source,
        conversation_id=row_id(row),
        model=clean_text(row.get("model")),
        language=clean_text(row.get("language")),
        domain=infer_domain(context),
        initial_prompt=prompt,
        first_response=messages[1]["content"],
        remaining_messages=list(messages[2:]),
        full_messages=list(messages),
    )

    return candidate, "accepted"


def stream_rows(source: str, split: str, hf_token: Optional[str]) -> Iterable[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if hf_token:
        kwargs["token"] = hf_token
    return load_dataset(source, **kwargs)


def load_used_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    used: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                used.update(record.get("prompt_hashes", []))
            except (json.JSONDecodeError, TypeError):
                continue
    return used


from collections import Counter


def collect_candidates(
    source: str,
    split: str,
    hf_token: Optional[str],
    used_hashes: set[str],
    scan_limit: int,
    interactive_limit: int,
    single_limit: int,
    adapter,
) -> Tuple[List[Candidate], List[Candidate]]:

    interactive: List[Candidate] = []
    single: List[Candidate] = []

    seen_local: set[str] = set()
    diagnostics = Counter()

    rows = stream_rows(
        source=source,
        split=split,
        hf_token=hf_token,
    )

    for index, row in enumerate(rows):
        if index >= scan_limit:
            break

        diagnostics["rows_seen"] += 1

        candidate, reason = row_to_candidate(
            row=row,
            source=source,
            adapter=adapter,
        )

        diagnostics[reason] += 1

        if candidate is None:
            continue

        prompt_hash = text_hash(
            candidate.initial_prompt
        )

        if prompt_hash in used_hashes:
            diagnostics["previously_used"] += 1
            continue

        if prompt_hash in seen_local:
            diagnostics["duplicate_in_current_run"] += 1
            continue

        seen_local.add(prompt_hash)

        messages = candidate.full_messages

        clarification = is_clarification(
            candidate.first_response
        )

        if len(messages) >= 4:
            diagnostics["multi_turn"] += 1

            if clarification:
                diagnostics["clarification_detected"] += 1

                user_reply = clean_text(
                    messages[2]["content"]
                )

                if (
                        messages[2]["role"] == "user"
                        and len(
                            user_reply.split()
                        ) <= 20
                ):
                    interactive.append(candidate)
                    diagnostics["interactive_added"] += 1
            else:
                diagnostics["multi_turn_no_clarification"] += 1

        elif len(messages) == 2:
            diagnostics["two_message"] += 1

            if not clarification:
                single.append(candidate)
                diagnostics["single_added"] += 1
            else:
                diagnostics[
                    "two_message_clarification"
                ] += 1

        else:
            diagnostics["other_message_count"] += 1

        if (
            diagnostics["rows_seen"] % 10_000 == 0
        ):
            logging.info(
                f"Rows={diagnostics['rows_seen']} "
                f"Accepted={diagnostics['accepted']} "
                f"Interactive={len(interactive)} "
                f"Single={len(single)}"
            )

        if (
            len(interactive) >= interactive_limit
            and len(single) >= single_limit
        ):
            diagnostics["candidate_limits_reached"] += 1
            break

    print()
    print("Candidate collection diagnostics")
    print("=" * 55)

    for name, value in diagnostics.most_common():
        print(f"{name:40s}: {value:,}")

    print("=" * 55)
    print(
        f"Interactive candidates: {len(interactive):,}"
    )
    print(
        f"Single-shot candidates: {len(single):,}"
    )

    json.dump(
        diagnostics,
        open(
            "Data/candidate_stats.json",
            "w",
            encoding="utf-8"
        ),
        indent=2
    )

    return interactive, single


def match_pairs_tfidf(
    interactive: Sequence[Candidate],
    single: Sequence[Candidate],
    n: int,
    min_similarity: float,
    domain_bonus: float,
) -> List[Pair]:
    if not interactive or not single:
        return []

    corpus = [c.initial_prompt for c in interactive] + [c.initial_prompt for c in single]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=100_000,
    )
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[: len(interactive)], matrix[len(interactive) :])

    ranked: List[Tuple[float, float, int, int]] = []
    for i, left in enumerate(interactive):
        for j, right in enumerate(single):
            raw = float(sims[i, j])
            if raw < min_similarity:
                continue
            bonus = domain_bonus if left.domain == right.domain else 0.0
            ranked.append((raw + bonus, raw, i, j))
    ranked.sort(reverse=True)

    output: List[Pair] = []
    used_i: set[int] = set()
    used_j: set[int] = set()
    domain_counts: Dict[str, int] = {}

    # Greedy one-to-one matching, with a soft diversity penalty.
    while ranked and len(output) < n:
        best_pos = max(
            range(len(ranked)),
            key=lambda k: ranked[k][0] - 0.03 * domain_counts.get(interactive[ranked[k][2]].domain, 0),
        )
        _, raw, i, j = ranked.pop(best_pos)
        if i in used_i or j in used_j:
            continue
        output.append(Pair(interactive[i], single[j], raw))
        used_i.add(i)
        used_j.add(j)
        d = interactive[i].domain
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return output

def match_pairs_embeddings(
    interactive: Sequence[Candidate],
    single: Sequence[Candidate],
    n: int,
    min_similarity: float,
    domain_bonus: float,
) -> List:
    model = get_embedding_model()

    if not interactive or not single:
        return []

    interactive_prompts = [
        c.initial_prompt
        for c in interactive
    ]

    single_prompts = [
        c.initial_prompt
        for c in single
    ]

    print(
        "Generating embeddings..."
    )

    interactive_embeddings = model.encode(
        interactive_prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    single_embeddings = model.encode(
        single_prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sims = cosine_similarity(
        interactive_embeddings,
        single_embeddings,
    )

    ranked: List[
        Tuple[float, float, int, int]
    ] = []

    for i, left in enumerate(interactive):
        for j, right in enumerate(single):

            raw = float(
                sims[i, j]
            )

            if raw < min_similarity:
                continue

            bonus = (
                domain_bonus
                if left.domain == right.domain
                else 0.0
            )

            ranked.append(
                (
                    raw + bonus,
                    raw,
                    i,
                    j,
                )
            )

    ranked.sort(
        reverse=True
    )

    output: List[Pair] = []

    used_i: set[int] = set()
    used_j: set[int] = set()

    domain_counts: Dict[str, int] = {}

    while (
        ranked
        and len(output) < n
    ):

        best_pos = max(
            range(len(ranked)),
            key=lambda k:
                ranked[k][0]
                -
                0.03 *
                domain_counts.get(
                    interactive[
                        ranked[k][2]
                    ].domain,
                    0
                )
        )

        _, raw, i, j = ranked.pop(
            best_pos
        )

        if (
            i in used_i
            or j in used_j
        ):
            continue

        output.append(
            Pair(
                interactive[i],
                single[j],
                raw,
            )
        )

        used_i.add(i)
        used_j.add(j)

        domain = interactive[i].domain

        domain_counts[domain] = (
            domain_counts.get(
                domain,
                0
            ) + 1
        )

    return output

def match_pairs(
    interactive,
    single,
    n,
    min_similarity,
    domain_bonus,
    matching_method="tfidf",
):
    if matching_method == "tfidf":
        return match_pairs_tfidf(
            interactive,
            single,
            n,
            min_similarity,
            domain_bonus,
        )

    if matching_method == "embeddings":
        return match_pairs_embeddings(
            interactive,
            single,
            n,
            min_similarity,
            domain_bonus,
        )

    raise ValueError(
        f"Unknown matching method: "
        f"{matching_method}"
    )


def append_results(pairs: Sequence[Pair], history_path: Path, output_path: Path, matching_method: str) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as history, output_path.open("a", encoding="utf-8") as out:
        for pair in pairs:
            hashes = [
                text_hash(pair.interactive.initial_prompt),
                text_hash(pair.single_shot.initial_prompt),
            ]
            history.write(json.dumps({"prompt_hashes": hashes}, ensure_ascii=False) + "\n")
            record = {
                "matching_method": matching_method,
                "similarity": round(pair.similarity, 6),
                "interactive": asdict(pair.interactive),
                "single_shot": asdict(pair.single_shot),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(
    source: str = "allenai/WildChat",
    split: str = "train",
    n: int = 10,
    history_file: str = "selected_prompt_history.jsonl",
    output_file: str = "matched_examples.jsonl",
    matching_method: str = "tfidf",
    scan_limit: int = 150_000,
    interactive_limit: int = 1_500,
    single_limit: int = 8_000,
    min_similarity: float = 0.16,
    domain_bonus: float = 0.04,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return up to n new pairs and append them to persistent JSONL files."""
    history_path = Path(history_file)
    output_path = Path(output_file)
    used_hashes = load_used_hashes(history_path)

    adapter = get_adapter(source)


    interactive, single = collect_candidates(
        source=source,
        split=split,
        hf_token=hf_token,
        used_hashes=used_hashes,
        scan_limit=scan_limit,
        interactive_limit=interactive_limit,
        single_limit=single_limit,
        adapter=adapter,
    )
    print("Interactive:", len(interactive))
    print("Single-shot:", len(single))
    pairs = match_pairs(
        interactive,
        single,
        n,
        min_similarity,
        domain_bonus,
        matching_method,
    )
    append_results(pairs, history_path, output_path, matching_method)

    results = [
        {
            "similarity": round(p.similarity, 6),
            "domain": p.interactive.domain,
            "interactive_prompt": p.interactive.initial_prompt,
            "clarifying_response": p.interactive.first_response,
            "follow_up_messages": p.interactive.remaining_messages,
            "single_shot_prompt": p.single_shot.initial_prompt,
            "single_shot_response": p.single_shot.first_response,
            "interactive_id": p.interactive.conversation_id,
            "single_shot_id": p.single_shot.conversation_id,
        }
        for p in pairs
    ]
    return results


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="allenai/WildChat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--history-file", default="Data/selected_prompt_history.jsonl")
    parser.add_argument("--output-file", default="Data/matched_examples.jsonl")
    parser.add_argument("--matching-method", choices=['tfidf', 'embeddings'], default="tfidf")
    parser.add_argument("--scan-limit", type=int, default=150_000)
    parser.add_argument("--interactive-limit", type=int, default=1_500)
    parser.add_argument("--single-limit", type=int, default=8_000)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--domain-bonus", type=float, default=0.04)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if args.min_similarity is None:

        if args.matching_method == "tfidf":
            args.min_similarity = 0.16

        elif args.matching_method == "embeddings":
            args.min_similarity = 0.75

    logging.info(
        f"Matching method={args.matching_method} "
        f"min_similarity={args.min_similarity}"
    )

    results = main(
        source=args.source,
        split=args.split,
        n=args.n,
        history_file=args.history_file,
        output_file=args.output_file,
        matching_method=args.matching_method,
        scan_limit=args.scan_limit,
        interactive_limit=args.interactive_limit,
        single_limit=args.single_limit,
        min_similarity=args.min_similarity,
        domain_bonus=args.domain_bonus,
        hf_token=args.hf_token,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\\nReturned {len(results)} new pairs.")


if __name__ == "__main__":
    cli()
