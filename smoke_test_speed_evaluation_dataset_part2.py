#!/usr/bin/env python3
"""Smoke-test and benchmark generation speed on part-2 evaluation data.

For each model and domain, throughput is computed as:

    total generated tokens / total wall-clock time spent inside model.generate()

Tokenization and prompt preparation are excluded. CUDA is synchronized immediately
before and after each timed generate() call so asynchronous GPU work is included.
"""

import argparse
import json
import platform
import re
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
DEFAULT_DATASET = "speed_evaluation_dataset_part2.json"
DEFAULT_REPORT = "smoke_test_speed_evaluation_dataset_part2_report.md"
DOMAINS = ("agentic_coding", "tool_calling")


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


def normalize_messages(raw_messages: Any) -> list[dict]:
    messages = parse_jsonish(raw_messages)
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized = []
    for raw in messages:
        raw = parse_jsonish(raw)
        if not isinstance(raw, dict):
            raise ValueError("Message must be an object")
        role = raw.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role {role!r}")
        content = raw.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        msg = {"role": role, "content": content}
        calls = parse_jsonish(raw.get("tool_calls", []))
        if calls:
            if isinstance(calls, dict):
                calls = [calls]
            msg["tool_calls"] = calls
        for key in ("tool_call_id", "name"):
            if raw.get(key) is not None:
                msg[key] = raw[key]
        normalized.append(msg)
    return normalized


def extract_json_objects(text: str) -> list[dict]:
    chunks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    if not chunks:
        chunks = [text]
    decoder = json.JSONDecoder()
    objects = []
    for chunk in chunks:
        for match in re.finditer(r"\{", chunk):
            try:
                obj, _ = decoder.raw_decode(chunk[match.start():])
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
    return objects


def assess_tool_call(text: str, available_names: set[str]) -> dict:
    objects = extract_json_objects(text)
    calls = []
    for obj in objects:
        call = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = call.get("name")
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if name:
            calls.append({
                "name": name,
                "name_declared": name in available_names,
                "arguments_is_object": isinstance(args, dict),
            })
    return {
        "tool_marker_seen": any(marker in text for marker in (
            "<tool_call>", "<|python_tag|>", "<|tool_call_start|>"
        )),
        "json_objects_found": len(objects),
        "calls_found": calls,
        "valid_call_found": any(
            call["name_declared"] and call["arguments_is_object"] for call in calls
        ),
    }


def get_target_index(sample: dict) -> int:
    if sample.get("domain") == "tool_calling":
        return int(sample["validation"]["target_message_index"])
    return int(sample["metadata"]["target_message_index"])


def get_sample_id(sample: dict) -> str:
    if sample.get("domain") == "tool_calling":
        return str(sample.get("validation", {}).get("source_id", "unknown"))
    return str(sample.get("metadata", {}).get("session_id", "unknown"))


def prepare_examples(samples: list[dict], tokenizer, n_per_domain: int,
                     max_input_tokens: int) -> tuple[dict[str, list[dict]], dict[str, int]]:
    prepared = defaultdict(list)
    skipped = defaultdict(int)
    for sample in samples:
        domain = sample.get("domain", "unknown")
        if domain not in DOMAINS:
            skipped["unknown_domain"] += 1
            continue
        if len(prepared[domain]) >= n_per_domain:
            continue
        try:
            messages = normalize_messages(sample["messages"])
            target_index = get_target_index(sample)
            if not (0 < target_index < len(messages)):
                skipped[f"{domain}:bad_target_index"] += 1
                continue
            target = messages[target_index]
            if target["role"] != "assistant":
                skipped[f"{domain}:target_not_assistant"] += 1
                continue
            prompt_messages = messages[:target_index]
            if not prompt_messages or prompt_messages[-1]["role"] == "assistant":
                skipped[f"{domain}:prompt_ends_with_assistant"] += 1
                continue

            template_kwargs = {}
            tools = sample.get("tools") if domain == "tool_calling" else None
            if tools:
                template_kwargs["tools"] = tools
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > max_input_tokens:
                skipped[f"{domain}:over_context_limit"] += 1
                continue

            validation = sample.get("validation", {})
            prepared[domain].append({
                "domain": domain,
                "sample_id": get_sample_id(sample),
                "prompt": prompt_text,
                "input_tokens": input_tokens,
                "reference": target,
                "tools": tools or [],
                "available_tool_names": set(validation.get("available_tool_names", [])),
                "expected_tool_names": validation.get("expected_tool_names", []),
            })
        except Exception as exc:
            skipped[f"{domain}:{type(exc).__name__}"] += 1
    return prepared, dict(skipped)


def sync_cuda_if_needed(model) -> None:
    """Wait for queued GPU kernels before/after timing generate()."""
    if torch.cuda.is_available():
        # Synchronize each visible CUDA device to include asynchronous work from
        # models that are sharded across multiple devices by device_map="auto".
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def gpu_description() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable (CPU or non-CUDA backend)"
    descriptions = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        descriptions.append(
            f"GPU {index}: {props.name}, {props.total_memory / (1024 ** 3):.1f} GiB"
        )
    return "; ".join(descriptions)


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Maximum examples per domain and model")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help=f"Markdown report path (default: {DEFAULT_REPORT})")
    parser.add_argument("--show-output-chars", type=int, default=1200,
                        help="Maximum generated/reference text characters printed per sample; 0 disables")
    args = parser.parse_args()
    if args.num_examples < 1 or args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("--num-examples, --max-input-tokens and --max-new-tokens must be positive")

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("Expected the dataset JSON root to be a list")

    source_counts = Counter(sample.get("domain", "unknown") for sample in samples)
    print(f"Loaded {len(samples)} samples from {dataset_path}")
    print("Source domain counts:", dict(source_counts))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "dataset_samples": len(samples),
        "source_domain_counts": dict(source_counts),
        "models": [],
    }

    for model_name in args.models:
        print(f"\n{'=' * 90}\nMODEL: {model_name}")
        tokenizer = model = None
        model_record = {
            "model": model_name,
            "status": "failed",
            "domains": {},
            "skipped": {},
            "error": None,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            examples_by_domain, skipped = prepare_examples(
                samples, tokenizer, args.num_examples, args.max_input_tokens
            )
            model_record["skipped"] = skipped
            print("Prepared per domain:", {d: len(examples_by_domain.get(d, [])) for d in DOMAINS})
            print("Skipped:", skipped)

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            model.eval()
            model_record["status"] = "completed"

            for domain in DOMAINS:
                examples = examples_by_domain.get(domain, [])
                domain_record = {
                    "examples_prepared": len(examples),
                    "examples_succeeded": 0,
                    "examples_failed": 0,
                    "total_input_tokens": 0,
                    "total_generated_tokens": 0,
                    "total_generation_seconds": 0.0,
                    "tokens_per_second": None,
                    "samples": [],
                }
                print(f"\n### DOMAIN: {domain} ({len(examples)} examples)")
                if not examples:
                    print("[WARN] No compatible examples fit the context budget.")
                    model_record["domains"][domain] = domain_record
                    continue

                for idx, example in enumerate(examples, 1):
                    print(
                        f"\n--- {idx}/{len(examples)} | id={example['sample_id']} "
                        f"| input_tokens={example['input_tokens']} ---"
                    )
                    sample_record = {
                        "sample_id": example["sample_id"],
                        "input_tokens": example["input_tokens"],
                        "generated_tokens": 0,
                        "generation_seconds": None,
                        "tokens_per_second": None,
                        "status": "failed",
                        "error": None,
                        "tool_call_assessment": None,
                    }
                    try:
                        encoded = tokenizer(
                            example["prompt"], return_tensors="pt", add_special_tokens=False
                        ).to(model.device)
                        input_length = int(encoded["input_ids"].shape[-1])

                        # Match eval_speed_qwen.py: time the generate call, not tokenization.
                        sync_cuda_if_needed(model)
                        start_time = time.perf_counter()
                        with torch.inference_mode():
                            generated = model.generate(
                                **encoded,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=(tokenizer.pad_token_id
                                              if tokenizer.pad_token_id is not None
                                              else tokenizer.eos_token_id),
                            )
                        sync_cuda_if_needed(model)
                        elapsed = time.perf_counter() - start_time

                        new_ids = generated[0, input_length:]
                        # generate() normally stops at EOS. Count the returned sequence tokens,
                        # including EOS if present, consistently with output_ids slicing in reference.
                        generated_tokens = int(new_ids.numel())
                        sample_tps = generated_tokens / elapsed if elapsed > 0 else None
                        output = tokenizer.decode(new_ids, skip_special_tokens=False)

                        sample_record.update({
                            "generated_tokens": generated_tokens,
                            "generation_seconds": elapsed,
                            "tokens_per_second": sample_tps,
                            "status": "success",
                        })
                        if domain == "tool_calling":
                            assessment = assess_tool_call(output, example["available_tool_names"])
                            assessment["expected_tool_names"] = example["expected_tool_names"]
                            sample_record["tool_call_assessment"] = assessment
                            print("Tool-call assessment:", json.dumps(assessment, ensure_ascii=False))

                        domain_record["examples_succeeded"] += 1
                        domain_record["total_input_tokens"] += input_length
                        domain_record["total_generated_tokens"] += generated_tokens
                        domain_record["total_generation_seconds"] += elapsed
                        print(f"Generated tokens: {generated_tokens}")
                        print(f"model.generate() wall time: {elapsed:.4f} s")
                        print(f"Sample throughput: {sample_tps:.2f} tokens/s" if sample_tps is not None else "Sample throughput: n/a")
                        if args.show_output_chars:
                            print("Generated output:\n", output[:args.show_output_chars])
                            print("Reference target:\n", json.dumps(
                                example["reference"], ensure_ascii=False
                            )[:args.show_output_chars])
                        print("[PASS] Generation ran." if generated_tokens else "[WARN] Empty generation.")
                    except Exception as exc:
                        sample_record["error"] = f"{type(exc).__name__}: {exc}"
                        domain_record["examples_failed"] += 1
                        print(f"[GENERATION FAIL] {type(exc).__name__}: {exc}")
                        traceback.print_exc(limit=2)
                    domain_record["samples"].append(sample_record)

                total_seconds = domain_record["total_generation_seconds"]
                total_tokens = domain_record["total_generated_tokens"]
                domain_record["tokens_per_second"] = (
                    total_tokens / total_seconds if total_seconds > 0 else None
                )
                model_record["domains"][domain] = domain_record
                tps = domain_record["tokens_per_second"]
                print(
                    f"\nDOMAIN TOTAL {domain}: {total_tokens} generated tokens / "
                    f"{total_seconds:.4f} s = "
                    f"{tps:.2f} tokens/s" if tps is not None else
                    f"\nDOMAIN TOTAL {domain}: no successful timed generations"
                )
        except Exception as exc:
            model_record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[MODEL/FORMAT FAIL] {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
        finally:
            report["models"].append(model_record)
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_report(Path(args.report), report, args)
    print(f"\nMarkdown report written to: {args.report}")


def write_report(report_path: Path, report: dict, args) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Part 2 Speed Evaluation Report",
        "",
        f"- **Generated:** {report['created_utc']} (UTC)",
        f"- **Dataset:** `{report['dataset']}`",
        f"- **Dataset records:** {report['dataset_samples']}",
        f"- **Python:** `{platform.python_version()}`",
        f"- **PyTorch:** `{torch.__version__}`",
        f"- **Transformers:** `{transformers.__version__}`",
        f"- **CUDA available:** `{torch.cuda.is_available()}`",
        f"- **CUDA runtime reported by PyTorch:** `{torch.version.cuda}`",
        f"- **Hardware:** {gpu_description()}",
        f"- **Requested examples per domain/model:** {args.num_examples}",
        f"- **Maximum input length:** {args.max_input_tokens} tokens",
        f"- **Generation limit:** {args.max_new_tokens} new tokens per sample",
        "- **Decoding:** greedy (`do_sample=False`)",
        "",
        "## Measurement methodology",
        "",
        "For each model and domain, throughput is calculated as **the sum of generated output tokens divided by the sum of `model.generate()` wall-clock seconds** across successful samples. This is a weighted aggregate, not the arithmetic mean of per-sample tokens/second.",
        "",
        "Prompt formatting and tokenization are performed outside the timed interval. CUDA is synchronized immediately before and after each generation call, so asynchronous GPU work is included in elapsed time. Model loading, warm-up (none is performed explicitly), and report writing are excluded. Results are specific to the tested hardware, model, Transformers/PyTorch versions, prompt subset, and generation limit.",
        "",
        "The returned generated sequence is counted after removing the input prefix. The token count includes any EOS token present in the returned sequence. Failed generations are excluded from token and time totals and listed separately.",
        "",
        "## Source dataset counts",
        "",
        "| Domain | Records in dataset |",
        "|---|---:|",
    ]
    for domain in DOMAINS:
        lines.append(f"| `{domain}` | {report['source_domain_counts'].get(domain, 0)} |")
    lines += ["", "## Domain throughput summary", "", "One row per model and domain.", "",
              "| Model | Domain | Prepared | Succeeded | Failed | Generated tokens | Generate time (s) | Aggregate throughput (tokens/s) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        for domain in DOMAINS:
            d = model["domains"].get(domain, {})
            tps = d.get("tokens_per_second")
            lines.append(
                f"| `{md_escape(model['model'])}` | `{domain}` | "
                f"{d.get('examples_prepared', 0)} | {d.get('examples_succeeded', 0)} | "
                f"{d.get('examples_failed', 0)} | {d.get('total_generated_tokens', 0)} | "
                f"{d.get('total_generation_seconds', 0.0):.4f} | "
                f"{tps:.2f} |" if tps is not None else
                f"| `{md_escape(model['model'])}` | `{domain}` | "
                f"{d.get('examples_prepared', 0)} | {d.get('examples_succeeded', 0)} | "
                f"{d.get('examples_failed', 0)} | {d.get('total_generated_tokens', 0)} | "
                f"{d.get('total_generation_seconds', 0.0):.4f} | n/a |"
            )
    lines += ["", "## Per-sample measurements", ""]
    for model in report["models"]:
        lines += [f"### `{md_escape(model['model'])}`", "", f"- **Model status:** {model['status']}"]
        if model.get("error"):
            lines.append(f"- **Model error:** `{md_escape(model['error'])}`")
        if model.get("skipped"):
            lines.append("- **Skipped samples:** " + ", ".join(
                f"`{md_escape(k)}`: {v}" for k, v in sorted(model["skipped"].items())
            ))
        lines.append("")
        for domain in DOMAINS:
            d = model["domains"].get(domain, {})
            lines += [f"#### {domain}", "",
                      "| Sample ID | Status | Input tokens | Generated tokens | `generate()` time (s) | Sample throughput (tokens/s) | Details |",
                      "|---|---|---:|---:|---:|---:|---|"]
            for sample in d.get("samples", []):
                elapsed = sample.get("generation_seconds")
                tps = sample.get("tokens_per_second")
                details = sample.get("error") or ""
                assessment = sample.get("tool_call_assessment")
                if assessment is not None:
                    details = (
                        f"valid tool call: {assessment['valid_call_found']}; "
                        f"calls found: {len(assessment['calls_found'])}; "
                        f"expected: {', '.join(assessment.get('expected_tool_names', []))}"
                    )
                lines.append(
                    f"| `{md_escape(sample.get('sample_id', 'unknown'))}` | {sample.get('status')} | "
                    f"{sample.get('input_tokens', 0)} | {sample.get('generated_tokens', 0)} | "
                    f"{elapsed:.4f} | {tps:.2f} | {md_escape(details)} |"
                    if elapsed is not None and tps is not None else
                    f"| `{md_escape(sample.get('sample_id', 'unknown'))}` | {sample.get('status')} | "
                    f"{sample.get('input_tokens', 0)} | {sample.get('generated_tokens', 0)} | n/a | n/a | "
                    f"{md_escape(details)} |"
                )
            if not d.get("samples"):
                lines.append("| — | no samples prepared | 0 | 0 | n/a | n/a | — |")
            lines.append("")
    lines += [
        "## Interpretation and limitations",
        "",
        "- This script is a small smoke-test benchmark, not a statistically rigorous full-dataset benchmark. Increase `--num-examples` for more stable measurements.",
        "- The default model order and examples may introduce warm-up/cache effects. For careful comparisons, use the same settings and consider a separate warm-up run before collecting results.",
        "- The tool-calling domain measures generation of a tool-call-like assistant response; it does not execute the requested tool or measure a complete agent loop.",
        "- Agentic-coding and tool-calling prompts have different formats and lengths. Compare models within each domain, and avoid interpreting cross-domain differences as pure model speed differences.",
        "- Throughput uses generated output tokens only; prompt-processing time is not included.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
