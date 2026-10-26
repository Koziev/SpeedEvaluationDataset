#!/usr/bin/env python3
"""Build speed_evaluation_dataset_part2.json with agentic-coding and tool-calling samples.

Dependencies:
    pip install -U datasets

The output is a JSON list. Each item has `domain` and the complete normalized
`messages` trajectory. `metadata.target_message_index` identifies the assistant
message to predict; the smoke test uses only messages before that index as input.

Tool-calling samples additionally preserve `tools` and a `validation` object
containing the expected tool call(s) and available tool names.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


AGENTIC_DATASET = "thoughtworks/agentic-coding-trajectories"
TOOL_DATASET = "Archangel-system/glaive-function-calling-v2-openai-native"

# `max_isl` uses cl100k_base token counts, not Qwen tokenizer counts.
ISL_BANDS = [
    (0, 8_000),
    (8_000, 16_000),
    (16_000, 24_000),
    (24_000, 32_000),
    (32_000, 64_000),
    (64_000, 10**12),
]


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_tool_calls(value: Any) -> list[dict]:
    calls = parse_jsonish(value)
    if calls is None or calls == "":
        return []
    if isinstance(calls, dict):
        if "tool_calls" in calls:
            calls = calls["tool_calls"]
        else:
            calls = [calls]
    if not isinstance(calls, list):
        raise ValueError(f"Expected tool_calls list/dict, got {type(calls).__name__}")

    result = []
    for raw in calls:
        call = parse_jsonish(raw)
        if not isinstance(call, dict):
            raise ValueError("Tool-call entry is not an object")
        call = dict(call)
        function = call.get("function")
        if isinstance(function, dict):
            function = dict(function)
            if "arguments" in function:
                function["arguments"] = parse_jsonish(function["arguments"])
            call["function"] = function
        result.append(call)
    return result


def normalize_agentic_messages(value: Any) -> list[dict]:
    raw_messages = parse_jsonish(value)
    if not isinstance(raw_messages, list):
        raise ValueError("messages_json is not a JSON list")

    tool_names_by_id = {}
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        for call in normalize_tool_calls(raw.get("tool_calls_json")):
            name = (call.get("function") or {}).get("name") or call.get("name")
            if call.get("id") and name:
                tool_names_by_id[call["id"]] = name

    messages = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("Message entry is not an object")
        role = raw.get("role")
        if role == "developer":
            role = "system"
        elif role == "function":
            role = "tool"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role!r}")

        content = raw.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        message = {"role": role, "content": content}
        calls = normalize_tool_calls(raw.get("tool_calls_json"))
        if calls:
            message["tool_calls"] = calls

        call_id = raw.get("tool_call_id")
        if call_id is not None:
            message["tool_call_id"] = call_id
            if role == "tool" and call_id in tool_names_by_id:
                message["name"] = tool_names_by_id[call_id]
        elif raw.get("name"):
            message["name"] = raw["name"]

        messages.append(message)
    return messages


def normalize_glaive_messages(value: Any) -> list[dict]:
    raw_messages = parse_jsonish(value)
    if not isinstance(raw_messages, list):
        raise ValueError("Glaive messages field is not a JSON list")

    tool_names_by_id = {}
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        for call in normalize_tool_calls(raw.get("tool_calls")):
            name = (call.get("function") or {}).get("name") or call.get("name")
            if call.get("id") and name:
                tool_names_by_id[call["id"]] = name

    messages = []
    for raw in raw_messages:
        raw = parse_jsonish(raw)
        if not isinstance(raw, dict):
            raise ValueError("Message entry is not an object")
        role = raw.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported Glaive role: {role!r}")
        content = raw.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        message = {"role": role, "content": content}

        calls = normalize_tool_calls(raw.get("tool_calls"))
        if calls:
            message["tool_calls"] = calls
        if raw.get("tool_call_id"):
            message["tool_call_id"] = raw["tool_call_id"]
            if role == "tool" and raw["tool_call_id"] in tool_names_by_id:
                message["name"] = tool_names_by_id[raw["tool_call_id"]]
        if raw.get("name"):
            message["name"] = raw["name"]
        messages.append(message)
    return messages


def normalize_tools(value: Any) -> list[dict]:
    tools = parse_jsonish(value)
    if not isinstance(tools, list):
        raise ValueError(f"Expected tools list, got {type(tools).__name__}")
    normalized = []
    for raw in tools:
        tool = parse_jsonish(raw)
        if not isinstance(tool, dict):
            raise ValueError("Tool definition is not an object")
        tool = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            function = dict(function)
            if "parameters" in function:
                function["parameters"] = parse_jsonish(function["parameters"])
            tool["function"] = function
        normalized.append(tool)
    return normalized


def target_for_agentic(messages: list[dict]) -> int | None:
    """Prefer a final natural-language/code assistant turn following a user/tool turn."""
    candidates = [
        i for i, msg in enumerate(messages)
        if msg["role"] == "assistant"
        and (msg.get("content", "").strip() or msg.get("tool_calls"))
        and i > 0
        and messages[i - 1]["role"] in {"user", "tool"}
        and not msg.get("tool_calls")
    ]
    if candidates:
        return candidates[-1]

    # Fallback: a tool-call assistant turn can still be a useful coding-agent target.
    candidates = [
        i for i, msg in enumerate(messages)
        if msg["role"] == "assistant"
        and msg.get("tool_calls")
        and i > 0
        and messages[i - 1]["role"] in {"user", "tool"}
    ]
    return candidates[-1] if candidates else None


def target_for_tool_call(messages: list[dict]) -> int | None:
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls") and i > 0:
            # A tool call should be predicted from preceding context, not included in it.
            if messages[i - 1]["role"] in {"user", "tool"}:
                return i
    return None


def available_tool_names(tools: list[dict]) -> list[str]:
    names = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(function["name"])
    return sorted(set(names))


def sample_agentic_rows(rows, n: int, seed: int) -> list[dict]:
    """Deterministically sample across the corpus's published max-ISL bands."""
    buckets = defaultdict(list)
    rejected = Counter()
    for row in rows:
        try:
            messages = normalize_agentic_messages(row["messages_json"])
            target_index = target_for_agentic(messages)
            if target_index is None:
                rejected["no_suitable_assistant_target"] += 1
                continue
            max_isl = int(row.get("max_isl") or 0)
            item = {
                "domain": "agentic_coding",
                "messages": messages,
                "metadata": {
                    "source_dataset": AGENTIC_DATASET,
                    "session_id": row.get("session_id"),
                    "source_dataset_group": row.get("source_dataset"),
                    "source_id": row.get("source_id"),
                    "agent_framework": row.get("agent_framework"),
                    "recorded_model": row.get("recorded_model"),
                    "n_turns": row.get("n_turns"),
                    "max_isl_cl100k": max_isl,
                    "total_tokens_cl100k": row.get("total_tokens"),
                    "ground_truth_meta": parse_jsonish(
                        row.get("ground_truth_meta_json")
                    ),
                    "target_message_index": target_index,
                },
            }
            for lo, hi in ISL_BANDS:
                if lo <= max_isl < hi:
                    buckets[(lo, hi)].append(item)
                    break
        except Exception as exc:
            rejected[type(exc).__name__] += 1

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Round-robin across length bands; if one band is sparse, continue from others.
    selected = []
    band_keys = list(buckets.keys())
    rng.shuffle(band_keys)
    while len(selected) < n and any(buckets.values()):
        for key in band_keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop())
    if len(selected) < n:
        raise RuntimeError(
            f"Only found {len(selected)} valid agentic-coding sessions; requested {n}. "
            f"Rejection counts: {dict(rejected)}"
        )
    return selected


def sample_tool_call_rows(rows, n: int, seed: int, split: str) -> list[dict]:
    candidates = []
    rejected = Counter()
    for row_idx, row in enumerate(rows):
        try:
            meta = parse_jsonish(row.get("meta", {}))
            category = meta.get("category", "") if isinstance(meta, dict) else ""
            if category != "tool_call":
                continue
            messages = normalize_glaive_messages(row["messages"])
            tools = normalize_tools(row["tools"])
            target_index = target_for_tool_call(messages)
            if target_index is None or not tools:
                rejected["no_target_or_tools"] += 1
                continue

            target_calls = messages[target_index]["tool_calls"]
            expected_names = sorted({
                (call.get("function") or {}).get("name")
                for call in target_calls
                if isinstance(call.get("function"), dict)
                and (call.get("function") or {}).get("name")
            })
            names = available_tool_names(tools)
            candidates.append({
                "domain": "tool_calling",
                "messages": messages,
                "tools": tools,
                "validation": {
                    "source_dataset": TOOL_DATASET,
                    "source_split": split,
                    "source_id": str(
                        row.get("session_id") or row.get("id") or f"row-{row_idx}"
                    ),
                    "category": category,
                    "target_message_index": target_index,
                    "expected_tool_calls": target_calls,
                    "expected_tool_names": expected_names,
                    "available_tool_names": names,
                    "all_expected_tools_declared": all(
                        name in names for name in expected_names
                    ),
                },
            })
        except Exception as exc:
            rejected[type(exc).__name__] += 1

    if len(candidates) < n:
        raise RuntimeError(
            f"Only found {len(candidates)} valid tool-call samples; requested {n}. "
            f"Rejection counts: {dict(rejected)}"
        )
    random.Random(seed + 1).shuffle(candidates)
    return candidates[:n]



def sample_identifier(sample: dict) -> str:
    """Return a stable, human-readable source identifier for the Markdown report."""
    if sample["domain"] == "agentic_coding":
        metadata = sample.get("metadata", {})
        return str(metadata.get("session_id") or metadata.get("source_id") or "unknown")
    validation = sample.get("validation", {})
    return str(validation.get("source_id") or "unknown")


def target_message_index(sample: dict) -> int:
    if sample["domain"] == "agentic_coding":
        return int(sample["metadata"]["target_message_index"])
    return int(sample["validation"]["target_message_index"])


def render_token_ids(tokenizer, messages: list[dict], tools: list[dict] | None,
                     add_generation_prompt: bool) -> list[int]:
    """Return the actual 1-D token-ID sequence from apply_chat_template.

    Recent Transformers versions may return a BatchEncoding/dict (with both
    `input_ids` and `attention_mask`) unless return_dict=False is specified.
    Calling len() on that mapping returns the number of keys (usually 2), not
    the number of tokens. Extract `input_ids` explicitly and flatten a possible
    single-example batch dimension.
    """
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": False,
    }
    if tools:
        kwargs["tools"] = tools
    result = tokenizer.apply_chat_template(messages, **kwargs)

    # Be defensive across Transformers versions/configurations.
    if isinstance(result, dict) or hasattr(result, "keys"):
        if "input_ids" not in result:
            raise ValueError(
                "apply_chat_template returned a mapping without 'input_ids': "
                f"{list(result.keys())}"
            )
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()

    # Flatten only a singleton batch dimension; preserve a plain token list.
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], list)
    ):
        result = result[0]

    if not isinstance(result, list) or any(
        not isinstance(token_id, int) for token_id in result
    ):
        raise TypeError(
            "Expected a flat list of token IDs from apply_chat_template; "
            f"got {type(result).__name__}"
        )
    return result


def percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(__import__("math").ceil(p * len(ordered)) - 1)))
    return ordered[index]


def format_int(value) -> str:
    return "—" if value is None else f"{value:,}"


def build_statistics(samples: list[dict], model_names: list[str]) -> tuple[str, list[dict]]:
    """Tokenize actual serialized prompts and target turns for each model tokenizer.

    Target-turn length is measured as the token-count delta between:
      (a) the prefix serialized with add_generation_prompt=True, and
      (b) that same prefix plus the reference assistant target serialized normally.
    This includes the model's tool-call serialization where applicable.
    """
    tokenizers = {}
    for model_name in model_names:
        print(f"Loading tokenizer for statistics: {model_name}")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    rows = []
    for sample_index, sample in enumerate(samples):
        domain = sample["domain"]
        messages = sample["messages"]
        target_index = target_message_index(sample)
        tools = sample.get("tools") if domain == "tool_calling" else None
        prompt_messages = messages[:target_index]
        full_target_messages = messages[:target_index + 1]

        row = {
            "domain": domain,
            "sample_id": sample_identifier(sample),
            "message_count": len(messages),
            "prompt_message_count": len(prompt_messages),
            "target_role": messages[target_index]["role"],
            "source_context_tokens_cl100k": (
                sample.get("metadata", {}).get("max_isl_cl100k")
                if domain == "agentic_coding" else None
            ),
        }

        for model_name, tokenizer in tokenizers.items():
            prefix_ids = render_token_ids(
                tokenizer, prompt_messages, tools, add_generation_prompt=True
            )
            full_ids = render_token_ids(
                tokenizer, full_target_messages, tools, add_generation_prompt=False
            )
            input_length = len(prefix_ids)
            target_length = max(0, len(full_ids) - input_length)
            row[f"{model_name}|input_tokens"] = input_length
            row[f"{model_name}|target_tokens"] = target_length

        rows.append(row)

        if (sample_index + 1) % 25 == 0 or sample_index + 1 == len(samples):
            print(f"Tokenized {sample_index + 1}/{len(samples)} samples")

    lines = [
        "# Speed Evaluation Dataset — Part 2: Dataset Statistics",
        "",
        "This report describes `speed_evaluation_dataset_part2.json`, containing "
        "agentic-coding trajectories and tool-calling examples.",
        "",
        "## Dataset composition",
        "",
        "| Domain | Samples | Source dataset |",
        "|---|---:|---|",
    ]
    domain_counts = Counter(sample["domain"] for sample in samples)
    source_by_domain = {
        "agentic_coding": AGENTIC_DATASET,
        "tool_calling": TOOL_DATASET,
    }
    for domain in ("agentic_coding", "tool_calling"):
        lines.append(
            f"| `{domain}` | {domain_counts.get(domain, 0)} | "
            f"`{source_by_domain[domain]}` |"
        )

    lines.extend([
        "",
        "## Token-length summary",
        "",
        "Lengths are computed using each model's own tokenizer and chat template. "
        "`Input tokens` count the serialized conversation prefix through the target "
        "turn boundary, with the assistant generation prompt and (for tool calling) "
        "the tool definitions included. `Expected target tokens` are estimated as "
        "the token-count difference between that prefix and the serialized prefix "
        "plus the reference assistant target turn. This includes template control "
        "tokens and tool-call serialization, so it is more useful than tokenizing "
        "the message's plain-text `content` alone. Small boundary effects from "
        "tokenization/template serialization are possible.",
        "",
        "| Domain | Tokenizer/model | N | Input min | Input median | Input mean | Input p90 | Input max | Target min | Target median | Target mean | Target p90 | Target max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    def stats_for(domain: str, model_name: str, key_suffix: str):
        vals = [
            int(row[f"{model_name}|{key_suffix}"])
            for row in rows if row["domain"] == domain
        ]
        if not vals:
            return []
        import statistics
        return [
            len(vals), min(vals), int(statistics.median(vals)),
            round(statistics.mean(vals)), percentile(vals, 0.90), max(vals)
        ]

    for domain in ("agentic_coding", "tool_calling"):
        for model_name in model_names:
            input_stats = stats_for(domain, model_name, "input_tokens")
            target_stats = stats_for(domain, model_name, "target_tokens")
            if not input_stats:
                continue
            n = input_stats[0]
            cells = [
                f"`{domain}`", f"`{model_name}`", str(n),
                *(format_int(x) for x in input_stats[1:]),
                *(format_int(x) for x in target_stats[1:]),
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Per-sample token lengths",
        "",
        "One row per sample. `Input` and `target` lengths are tokenizer-specific. "
        "For agentic-coding samples, `Source max ISL` is the source dataset's "
        "`max_isl` value measured with `cl100k_base`; it is included for reference "
        "and is not directly comparable to Qwen token counts. ISL stands for *Input Sequence Length*.",
        "",
    ])

    for domain in ("agentic_coding", "tool_calling"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        lines.extend([
            f"### `{domain}`",
            "",
            "| Sample ID | Messages | Prompt messages | Source max ISL (cl100k) | " +
            " | ".join(
                f"{model_name} input | {model_name} target"
                for model_name in model_names
            ) + " |",
            "|---|---:|---:|---:|" +
            "---:|---:|" * len(model_names),
        ])
        for row in domain_rows:
            cells = [
                f"`{row['sample_id'].replace('|', '\\|')}`",
                str(row["message_count"]),
                str(row["prompt_message_count"]),
                format_int(row["source_context_tokens_cl100k"]),
            ]
            for model_name in model_names:
                cells.append(format_int(row[f"{model_name}|input_tokens"]))
                cells.append(format_int(row[f"{model_name}|target_tokens"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "## Reproducibility notes",
        "",
        "- Dataset sampling seed and source splits are set by the builder's CLI arguments.",
        "- Input lengths are measured after applying each tokenizer's chat template; "
        "tool definitions are passed to the template for `tool_calling` samples.",
        "- The expected target is identified by `metadata.target_message_index` "
        "for `agentic_coding` and `validation.target_message_index` for `tool_calling`.",
        "- These are reference-turn length statistics, not model-generated output lengths "
        "or speculative-decoding acceptance measurements.",
        "",
    ])
    return "\n".join(lines), rows



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="speed_evaluation_dataset_part2.json")
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agentic-split", default="train")
    parser.add_argument("--tool-split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--stats-models", nargs="+",
        default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"],
        help="Tokenizers used to report model-specific prompt and target lengths.",
    )
    args = parser.parse_args()

    print(f"Loading {AGENTIC_DATASET} ({args.agentic_split})...")
    agentic = load_dataset(AGENTIC_DATASET, split=args.agentic_split)
    print(f"Loaded {len(agentic):,} agentic-coding sessions")

    print(f"Loading {TOOL_DATASET} ({args.tool_split})...")
    tool_calling = load_dataset(TOOL_DATASET, split=args.tool_split)
    print(f"Loaded {len(tool_calling):,} tool-calling rows")

    agentic_samples = sample_agentic_rows(
        agentic, args.samples_per_domain, args.seed
    )
    tool_samples = sample_tool_call_rows(
        tool_calling, args.samples_per_domain, args.seed, args.tool_split
    )
    samples = agentic_samples + tool_samples
    random.Random(args.seed).shuffle(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    stats_markdown, _ = build_statistics(samples, args.stats_models)
    stats_path = output_path.with_suffix(".md")
    stats_path.write_text(stats_markdown, encoding="utf-8")
    print(f"Wrote statistics report to {stats_path.resolve()}")

    counts = Counter(sample["domain"] for sample in samples)
    print(f"\nWrote {len(samples)} samples to {output_path.resolve()}")
    print("Domain counts:", dict(counts))
    print("Agentic max-ISL bands (cl100k_base):")
    band_counts = Counter()
    for sample in agentic_samples:
        value = sample["metadata"]["max_isl_cl100k"]
        band = next((f"{lo}-{hi}" for lo, hi in ISL_BANDS if lo <= value < hi), "unknown")
        band_counts[band] += 1
    for band, count in sorted(band_counts.items()):
        print(f"  {band}: {count}")


if __name__ == "__main__":
    main()
