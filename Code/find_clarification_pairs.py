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

DIRECT_CLARIFICATION = re.compile(
    r"(?i)\b(?:"
    r"could you (?:please )?(?:provide|share|specify|clarify|"
    r"explain|elaborate|rephrase|complete|tell me)|"
    r"can you (?:please )?(?:provide|share|specify|clarify|"
    r"explain|elaborate|rephrase|complete|tell me)|"
    r"would you (?:please )?(?:provide|share|specify|clarify|"
    r"explain|elaborate|rephrase|complete|tell me)|"
    r"please (?:provide|share|specify|clarify|elaborate|"
    r"rephrase|complete|send|paste|upload)|"
    r"what do you mean(?: by)?|"
    r"do you mean|"
    r"which (?:one|version|language|framework|platform|"
    r"operating system|country|jurisdiction|format)|"
    r"what (?:version|language|framework|platform|"
    r"operating system|country|jurisdiction|format)|"
    r"are you referring to|"
    r"could you be more specific|"
    r"can you be more specific"
    r")\b"
)

MISSING_INFORMATION = re.compile(
    r"(?i)\b(?:"
    r"i need (?:more|additional|some) "
    r"(?:information|details|context)|"
    r"more information is needed|"
    r"more details are needed|"
    r"additional information is needed|"
    r"additional context is needed|"
    r"without (?:more|additional) "
    r"(?:information|details|context)|"
    r"not enough (?:information|details|context)|"
    r"the question is incomplete|"
    r"your question is incomplete|"
    r"it is not clear what you mean|"
    r"i(?:'m| am) not sure what you mean|"
    r"i(?:'m| am) not sure what you(?:'re| are) asking"
    r")\b"
)

GENERIC_CLOSING = re.compile(
    r"(?i)(?:"
    r"how can i (?:help|assist) you(?: today)?|"
    r"how may i (?:help|assist) you(?: today)?|"
    r"what can i (?:help|assist) you with(?: today)?|"
    r"is there anything else i can (?:help|assist) you with|"
    r"what else can i (?:help|assist) you with|"
    r"do you have any (?:other|further) questions|"
    r"let me know if you have any (?:other|further)? ?questions|"
    r"please let me know if you have any (?:other|further)? ?questions|"
    r"feel free to ask(?: me)?(?: any questions)?"
    r")[.!? ]*$"
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

SOURCE_NAMES = {
    "allenai/WildChat": "WildChat",
    "lmsys/lmsys-chat-1m": "LMSYS",
    "convai-challenge/conv_ai_3": "ClariQ",
    "sewon/ambig_qa": "AmbigQA",
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


def is_clarification(response: str) -> bool:
    text = clean_text(response)

    if not text:
        return False

    # Remove generic trailing offers before evaluating the response.
    substantive_text = GENERIC_CLOSING.sub("", text).strip()

    if not substantive_text:
        return False

    # Extremely long substantive answers are unlikely to be clarification
    # questions, even if they contain incidental trigger phrases.
    if len(substantive_text.split()) > 180:
        return False

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            substantive_text,
        )
        if sentence.strip()
    ]

    if not sentences:
        return False

    # A clarification request should appear near the end of the response.
    tail = " ".join(sentences[-2:])

    direct_request = bool(
        DIRECT_CLARIFICATION.search(tail)
    )

    missing_information = bool(
        MISSING_INFORMATION.search(tail)
    )

    asks_question = "?" in tail

    # A direct request is sufficient, even if punctuation is imperfect:
    # "Please provide the original text."
    if direct_request:
        return True

    # A statement of missing information must be paired with a question.
    # This prevents full answers containing incidental phrases from passing.
    if missing_information and asks_question:
        return True

    return False


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
    num_turns: int,
    single_limit: int,
    adapter,
    stats_path: str,
    language: str,
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

        if language:

            candidate_lang = (
                candidate.language
                .lower()
                .strip()
            )

            accepted = {
                language.lower(),
                "english",
            }

            if candidate_lang not in accepted:
                diagnostics["wrong_language"] += 1
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

        if len(messages) == 2:
            diagnostics["two_message"] += 1

            if clarification:
                diagnostics[
                    "two_message_clarification"
                ] += 1
                continue

            single.append(candidate)
            diagnostics["single_added"] += 1
            continue

        user_turns = sum(
            1
            for message in messages
            if message["role"] == "user"
        )

        if user_turns != num_turns:
            diagnostics["wrong_num_turns"] += 1
            continue

        diagnostics["multi_turn"] += 1

        if not clarification:
            diagnostics[
                "multi_turn_no_clarification"
            ] += 1
            continue

        diagnostics["clarification_detected"] += 1

        user_reply = clean_text(
            messages[2]["content"]
        )

        if messages[2]["role"] != "user":
            diagnostics[
                "missing_user_clarification"
            ] += 1
            continue

        if not user_reply:
            diagnostics[
                "empty_user_clarification"
            ] += 1
            continue

        if len(user_reply.split()) > 20:
            diagnostics[
                "long_user_clarification"
            ] += 1
            continue

        interactive.append(candidate)
        diagnostics["interactive_added"] += 1

        if diagnostics["clarification_detected"] <= 100:
            logging.info(
                "CLARIFICATION | prompt=%r | "
                "response=%r | user_reply=%r",
                candidate.initial_prompt,
                candidate.first_response,
                user_reply,
            )

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
            stats_path,
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
    language: str = "en",
    matching_method: str = "tfidf",
    scan_limit: int = 150_000,
    interactive_limit: int = 1_500,
    num_turns: int = 2,
    single_limit: int = 8_000,
    min_similarity: float = 0.16,
    domain_bonus: float = 0.04,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return up to n new pairs and append them to persistent JSONL files."""



    adapter = get_adapter(source)

    source_name = SOURCE_NAMES.get(
        source,
        source.replace("/", "_")
    )

    run_dir = (
            Path("Data")
            / source_name
            / matching_method
            / f"length_{num_turns}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path = (
            run_dir
            / "selected_prompt_history.jsonl"
    )

    output_path = (
            run_dir
            / "matched_examples.jsonl"
    )

    stats_path = (
            run_dir
            / "candidate_stats.json"
    )

    metadata_path = (
            run_dir
            / "run_metadata.json"
    )

    logging.basicConfig(
        filename=run_dir / "diagnostics.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
    )

    logging.getLogger(
        "httpx"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "httpcore"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "huggingface_hub"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "datasets"
    ).setLevel(logging.WARNING)

    used_hashes = load_used_hashes(history_path)

    interactive, single = collect_candidates(
        source=source,
        split=split,
        hf_token=hf_token,
        used_hashes=used_hashes,
        scan_limit=scan_limit,
        interactive_limit=interactive_limit,
        num_turns=num_turns,
        single_limit=single_limit,
        adapter=adapter,
        stats_path=stats_path,
        language=language,
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

    metadata = {
        "source": source,
        "source_name": source_name,
        "matching_method": matching_method,
        "scan_limit": scan_limit,
        "interactive_limit": interactive_limit,
        "num_turns": num_turns,
        "single_limit": single_limit,
        "min_similarity": min_similarity,
        "domain_bonus": domain_bonus,
        "generated_pairs": len(pairs),
        "language": language,
    }

    with metadata_path.open(
            "w",
            encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

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
    parser.add_argument("--language", default="en")
    parser.add_argument("--matching-method", choices=['tfidf', 'embeddings'], default="tfidf")
    parser.add_argument("--scan-limit", type=int, default=150_000)
    parser.add_argument("--interactive-limit", type=int, default=1_500)
    parser.add_argument("--num_turns", type=int, choices=[2, 3, 4, 5, 6], default=2)
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
        language=args.language,
        matching_method=args.matching_method,
        scan_limit=args.scan_limit,
        interactive_limit=args.interactive_limit,
        num_turns=args.num_turns,
        single_limit=args.single_limit,
        min_similarity=args.min_similarity,
        domain_bonus=args.domain_bonus,
        hf_token=args.hf_token,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\\nReturned {len(results)} new pairs.")


if __name__ == "__main__":
    cli()
