import collections
import datasets
import numpy as np
import terminaltables
import tqdm
import json
import traceback
import re
import os
import openpyxl
from openpyxl.utils import range_boundaries
import torch
import transformers
import unicodedata
import matplotlib.pyplot as plt
import gzip
import json
import requests


def is_russian_text(text: str) -> bool:
    """Perform fast check if provided text looks Russian (cyrillic, strictly speaking)."""

    # Set of Russian Cyrillic letters (including Ё/ё, which are outside the А-Я range)
    russian_letters = set(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    )

    # Keep only characters that are neither whitespace nor punctuation
    relevant = [
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    ]

    # Nothing meaningful to analyze
    if not relevant:
        return False

    russian_count = sum(1 for ch in relevant if ch in russian_letters)
    return russian_count / len(relevant) > 0.5


def is_chinese_text(text: str) -> bool:
    """Return True when Han characters make up most non-space, non-punctuation characters."""

    relevant = [
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    ]
    if not relevant:
        return False

    def is_han(ch: str) -> bool:
        cp = ord(ch)
        return (
            0x3400 <= cp <= 0x4DBF      # CJK Extension A
            or 0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
            or 0xF900 <= cp <= 0xFAFF   # CJK Compatibility Ideographs
            or 0x20000 <= cp <= 0x2FA1F # CJK Extensions B-F and related ranges
        )

    han_count = sum(1 for ch in relevant if is_han(ch))
    return han_count / len(relevant) > 0.5


def check_prompt_length(prompt: str) -> bool:
    if prompt is None or len(prompt) < 50 or len(prompt) > 4000:
        return False

    prompt_token_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    add_generation_prompt=True,
                                                    enable_thinking=False,
                                                    return_dict=True)["input_ids"]
    if len(prompt_token_ids) > MAX_PROMPT_TOKENS:
        return False

    return True


def check_chinese_prompt_length(prompt: str) -> bool:
    if prompt is None or len(prompt) < 10 or len(prompt) > 4000:
        return False

    prompt_token_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    add_generation_prompt=True,
                                                    enable_thinking=False,
                                                    return_dict=True)["input_ids"]
    if len(prompt_token_ids) > MAX_PROMPT_TOKENS:
        return False

    return True


def get_cell_range(file_path, cell_range, sheet_name=None):
    """Get values from a range of cells (e.g., 'A1:C5')"""
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True)

        if sheet_name:
            sheet = workbook[sheet_name]
        else:
            sheet = workbook.active

        # Parse the range
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)

        values = []
        for row in range(min_row, max_row + 1):
            row_values = []
            for col in range(min_col, max_col + 1):
                row_values.append(sheet.cell(row=row, column=col).value)
            values.append(row_values)

        return values

    except Exception as e:
        print(f"Error: {e}")
        return None


def build_lorugec():
    # LORuGEC
    samples = []
    fp = os.path.join("/home/inkoziev/polygon/text_generator/data/speller/LORuGEC/LORuGEC.xlsx")
    pairs = get_cell_range(fp, 'D2:G961')
    for i, (rule, _, corrupted, fixed) in enumerate(pairs):
        if rule in ("Punctuation",):
            continue

        corrupted = re.sub(r"[ ]{1,}([.,?!])", r"\1", corrupted)
        fixed = re.sub(r"[ ]{1,}([.,?!])", r"\1", fixed)
        if corrupted != fixed:
            # есть дефект
            samples.append({'text': corrupted, 'label': 1, 'domain': "LORuGEC"})

    return samples


def build_jfleg():
    raw = list(datasets.load_dataset('jhu-clsp/jfleg', split='test'))

    def convert(raw_samples):
        grammatical = []
        ungrammatical = []

        for sample in raw_samples:
            corrupted = sample['sentence']

            for corrected in sample['corrections']:
                if corrupted.lower() != corrected.lower():
                    grammatical.append({
                        'text': corrected,
                        'label': 0,
                        'domain': 'jfleg',
                    })

                    ungrammatical.append({
                        'text': corrupted,
                        'label': 1,
                        'domain': 'jfleg',
                    })

        return grammatical + ungrammatical

    return convert(raw)


def stop_token_ids(model, tokenizer) -> set:
    """Every id that ends a chat response: the generation config's EOS ids (what generate() stops on), the tokenizer's EOS
    and <|im_end|>. Counting up to the first of them also ignores the pad tokens that follow a finished row in a batch."""
    ids = set()
    for x in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id,
              tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if x is None:
            continue
        ids.update(x if isinstance(x, (list, tuple)) else [x])
    return {int(i) for i in ids if i is not None and i >= 0}


def stream_jsonl_gz(url):
    """Yield parsed JSON objects from a remote .jsonl.gz file without
    downloading it completely."""
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        # GzipFile reads the raw response stream and decompresses on the fly
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue



@torch.inference_mode()
def measure_response_lengths(model, tokenizer, prompts, batch_size: int, max_new_tokens: int, desc: str = ""):
    """Measure greedy response lengths, splitting OOM batches without recursive retries.

    Prompts that still OOM when generated individually are skipped rather than
    aborting the entire dataset build. Their response length is unknown, so they
    are excluded from the returned list.
    """
    stop_ids = stop_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = next(iter(stop_ids), tokenizer.eos_token_id)
    if pad_id is None:
        raise ValueError("Tokenizer must define a pad token or an EOS/stop token ID")

    encoded = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )["input_ids"]
        for p in prompts
    ]
    order = sorted(range(len(prompts)), key=lambda i: -len(encoded[i]))
    response_len = [None] * len(prompts)

    def try_generate(idx):
        """Return generated token IDs on CPU, or None on CUDA OOM.

        Keep this attempt in a separate stack frame. In particular, do not
        recursively retry from inside an except block: the OOM traceback can
        retain references to tensors from the failed generation while retries
        are running, preventing those allocations from being released.
        """
        input_ids = attention_mask = out = None
        try:
            L = max(len(encoded[i]) for i in idx)
            input_ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids = encoded[i]
                input_ids[r, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attention_mask[r, L - len(ids):] = 1

            out = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
            )
            # Copy generated tokens to CPU before releasing the GPU output.
            return out[:, L:].cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            del out, input_ids, attention_mask

    # Use an explicit work queue rather than recursive calls. Failed batch
    # frames are gone before smaller batches are attempted.
    pending = list(reversed([
        order[k:k + batch_size] for k in range(0, len(order), batch_size)
    ]))
    pbar = tqdm.tqdm(total=len(prompts), desc=desc)
    try:
        while pending:
            idx = pending.pop()
            generated = try_generate(idx)
            if generated is None:
                # The failed attempt's frame/traceback has now unwound, so the
                # allocator can release any newly-unused cached CUDA blocks.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(idx) > 1:
                    mid = len(idx) // 2
                    # Push second half first so the first half is processed next.
                    pending.append(idx[mid:])
                    pending.append(idx[:mid])
                else:
                    i = idx[0]
                    print(
                        f"WARNING: skipping prompt {i} in {desc!r}: CUDA OOM "
                        f"even with batch_size=1 (prompt_tokens={len(encoded[i])})."
                    )
                    pbar.update(1)
                continue

            for r, i in enumerate(idx):
                n = next(
                    (k + 1 for k, token_id in enumerate(generated[r]) if token_id in stop_ids),
                    len(generated[r]),
                )
                response_len[i] = min(n, max_new_tokens)
                pbar.update(1)
    finally:
        pbar.close()

    # OOM-skipped prompts have unknown lengths and must not be ranked as if
    # they had zero-length responses.
    return [
        (prompt, len(ids), n)
        for prompt, ids, n in zip(prompts, encoded, response_len)
        if n is not None
    ]


def plot_grouped_length_histograms(
    domain: str,
    data: dict,
    bins: int = 30,
    density: bool = False,
    gap: float = 0.2,        # fraction of bin width left empty between bins
    figsize=(10, 6),
    alpha: float = 0.75,
):
    """
    Grouped histograms of response lengths: shared bins, side-by-side bars,
    with a visible gap between adjacent bins.

    Parameters
    ----------
    domain : str
        Domain name
    data : dict[str, list[float]]
        Dictionary {language_code: [response_lengths]}.
    bins : int
        Number of bins (shared across all languages).
    density : bool
        Normalize each language's histogram (useful for unequal sample sizes).
    gap : float
        Fraction of the bin width left empty between bins (0..1).
        E.g. 0.2 means bars occupy 80% of the bin, 20% is a gap.
    figsize : tuple
        Figure size.
    alpha : float
        Bar opacity.
    """
    if not 0.0 <= gap < 1.0:
        raise ValueError("gap must be in [0, 1)")

    data = {lang: np.asarray(vals, dtype=float)
            for lang, vals in data.items() if len(vals) > 0}
    if not data:
        raise ValueError("No data to visualize")

    # Shared bin edges — identical for every language
    all_values = np.concatenate(list(data.values()))
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    lefts = bin_edges[:-1]

    langs = list(data.keys())
    n_langs = len(langs)

    # Width actually occupied by bars inside one bin (rest = gap)
    group_width = bin_width * (1.0 - gap)
    bar_width = group_width / n_langs

    # Offset of the group's left edge from the bin's left edge
    group_offset = (bin_width - group_width) / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    for i, lang in enumerate(langs):
        counts, _ = np.histogram(data[lang], bins=bin_edges, density=density)

        # Bars sit side-by-side, starting at group_offset inside the bin
        positions = lefts + group_offset + (i + 0.5) * bar_width

        ax.bar(
            positions,
            counts,
            width=bar_width,
            alpha=alpha,
            label=lang,
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xlabel("Response length, tokens")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Distribution of LLM response lengths in domain {domain}")
    ax.legend(title="Language")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    N_REPEATS = 100             # Number of prompts to keep per domain
    N_PREFETCH = 10 * N_REPEATS  # Number of prompts to sample from domain pool before ranking and selecting.
    GEN_BATCH_SIZE = 32         # prompts per generate() call; an OOM batch is split automatically. This setting is tuned for my RTX 3090 with 24 Gb of VRAM.
    MAX_NEW_TOKENS = 2048       # response-length probe; responses reaching it count as MAX_NEW_TOKENS
    MAX_PROMPT_TOKENS = MAX_NEW_TOKENS//2

    prompts = collections.defaultdict(list)

    shuffle_seed = 31415926

    model_id = "Qwen/Qwen3-0.6B"

    # ---------------------------------------------------------------------------

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")


    # Russian-English pairs
    print("Load pairs from Koziev/Translations...")
    URL = (
        "https://github.com/Koziev/Translations/raw/refs/heads/main/"
        "prose_translation_pairs.public.jsonl.gz"
    )
    for record in stream_jsonl_gz(URL):
        ru_text = record.get("right_text")
        if is_russian_text(ru_text) and len(ru_text) > 500:
            prompt = "Translate the text into English: {}".format(ru_text)
            if check_prompt_length(prompt):
                prompts["wmt ru-en"].append(prompt)

    ds_name = "/home/inkoziev/polygon/text_generator/tmp/sft_v053/sft23_dataset.unstressed.v053.eval.jsonl"
    print(f"Loading {ds_name}...")
    with open(ds_name) as f:
        for line in f:
            sample = json.loads(line)
            if sample["separator_token"] == "<poetry>":
                if len(sample["right_text"]) > 1000:
                    prompt = sample["left_text"]
                    prompts["poetry-ru"].append(prompt)
                    if len(prompts["poetry-ru"]) >= N_PREFETCH:
                        break

    # RUPOR dataset of GEC in different Russian text domains.
    print("Importing RUPOR...")
    src_fp = "/home/inkoziev/polygon/text_generator/py/generative_poetry/experiments/syntax_validation/Experiments4/datasets/RUPOR/rupor.json"
    gec_src_list = set()
    with open(src_fp) as f:
        data = json.load(f)
        for sample in data:
            if sample.get("corrupted_text") and sample.get("fixed_text") and sample["domain"] == "prose":
                gec_src_list.add(sample["corrupted_text"])

                if len(sample.get("fixed_text")) > 200:
                    prompts["wmt ru-en"].append("Translate the text into English: {}".format(sample.get("fixed_text")))

    print("Importing LoRuGEC...")
    lorugec = build_lorugec()
    for sample in lorugec:
        gec_src_list.add(sample["text"])

    gec_src_list = sorted(list(gec_src_list), key=lambda text: -len(text))
    gec_src_list = gec_src_list[:N_PREFETCH]
    for src in gec_src_list:
        prompt = "Исправь грамматические, пунктуационные и орфографические ошибки в тексте: {}".format(src)
        prompts["gec-ru"].append(prompt)

    print("Importing JFLEG...")
    jfleg = build_jfleg()
    gec_src_list = set(sample["text"] for sample in jfleg)
    gec_src_list = sorted(list(gec_src_list), key=lambda text: -len(text))
    gec_src_list = gec_src_list[:N_PREFETCH]
    for src in gec_src_list:
        prompt = "Correct grammar, punctuation, and spelling errors in the text: {}".format(src)
        prompts["gec-en"].append(prompt)


    # Chinese poetry-generation instructions. Filter a broad Chinese instruction
    # corpus for prompts specifically about composing poetry or classical verse.
    poetry_terms = ("写一首诗", "写首诗", "创作一首诗", "创作诗歌", "写一首古诗",
                    "作一首诗", "写诗", "诗歌创作", "填词", "写一首词", "对联",
                    "写一首歌", "写一首十四行诗", "诗")

    ds_name = "Iess/chinese_modern_poetry"
    print(f"Loading {ds_name} poetry candidates...")
    dataset = datasets.load_dataset(ds_name, split="train")
    for sample in tqdm.tqdm(dataset.shuffle(seed=shuffle_seed), desc=ds_name + " poetry"):
        q = str(sample.get("prompt", "")).strip()
        a = str(sample.get("response", "")).strip()
        if (
            q
            and a
            and is_chinese_text(q)
            and is_chinese_text(a)
            and any(term in q for term in poetry_terms)
            and check_chinese_prompt_length(q)
        ):
            prompts["poetry-zh"].append(q)
            if len(prompts["poetry-zh"]) >= N_PREFETCH:
                break

    if len(prompts["poetry-zh"]) < N_PREFETCH:
        ds_name = "lcccher/Chinese-Instruct"
        print(f"Loading {ds_name} poetry candidates...")
        dataset = datasets.load_dataset(ds_name, "firefly", split="train")
        for sample in tqdm.tqdm(dataset.shuffle(seed=shuffle_seed), desc=ds_name + " poetry"):
            q = str(sample.get("prompt", "")).strip()
            a = str(sample.get("response", "")).strip()
            if (
                q
                and a
                and is_chinese_text(q)
                and is_chinese_text(a)
                and any(term in q for term in poetry_terms)
                and check_chinese_prompt_length(q)
            ):
                prompts["poetry-zh"].append(q)
                if len(prompts["poetry-zh"]) >= N_PREFETCH:
                    break

    ds_name = "evilfreelancer/MATH-500-Russian"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="test")
    for sample in ds.shuffle(seed=shuffle_seed):
        prompts["math-ru"].append(sample["problem"])

    ds_name = "agentlans/allenai-WildChat-4.8M"
    dataset = datasets.load_dataset(ds_name, "Russian", split="train", streaming=False).shuffle(seed=shuffle_seed)
    for sample in tqdm.tqdm(dataset, desc=ds_name+" Russian"):
        messages = sample["conversations"]
        if len(messages) == 2 and messages[0]["from"] == "human" and messages[1]["from"] == "gpt":
            q = messages[0]["value"].strip()
            a = messages[1]["value"].strip()
            if check_prompt_length(q) and 200 < len(a) and is_russian_text(q) and is_russian_text(a):
                prompts["qa-ru"].append(q)
                if len(prompts["qa-ru"]) >= N_PREFETCH:
                    break

    dataset = datasets.load_dataset(ds_name, "English", split="train", streaming=False).shuffle(seed=shuffle_seed)
    for sample in tqdm.tqdm(dataset, desc=ds_name+" English"):
        messages = sample["conversations"]
        if len(messages) == 2 and messages[0]["from"] == "human" and messages[1]["from"] == "gpt":
            q = messages[0]["value"].strip()
            a = messages[1]["value"].strip()
            if check_prompt_length(q) and 200 < len(a):
                prompts["qa"].append(q)
                if len(prompts["qa"]) >= N_PREFETCH:
                    break

    # Chinese general question answering.
    ds_name = "agentlans/allenai-WildChat-4.8M"
    print(f"Loading {ds_name} Chinese...")
    dataset = datasets.load_dataset(ds_name, "Chinese", split="train", streaming=False).shuffle(seed=shuffle_seed)
    for sample in tqdm.tqdm(dataset, desc=ds_name + " Chinese"):
        messages = sample["conversations"]
        if len(messages) == 2 and messages[0]["from"] == "human" and messages[1]["from"] == "gpt":
            q = messages[0]["value"].strip()
            a = messages[1]["value"].strip()
            if (
                200 < len(q)
                and 200 < len(a)
                and is_chinese_text(q)
                and is_chinese_text(a)
                and check_chinese_prompt_length(q)
            ):
                prompts["qa-zh"].append(q)
                if len(prompts["qa-zh"]) >= N_PREFETCH:
                    break

    # Chinese mathematics. LiveK12Bench provides Chinese and English splits;
    # use only Chinese splits and retain math questions.
    ds_name = "Shawn-wxh/livek12bench"
    print(f"Loading {ds_name} Chinese mathematics...")
    livek12 = datasets.load_dataset(ds_name).shuffle(seed=shuffle_seed)
    for split_name in ("zh_2603", "zh_2605"):
        if split_name not in livek12:
            continue
        for sample in tqdm.tqdm(livek12[split_name], desc=f"{ds_name}:{split_name}"):
            if sample.get("subject") != "math":
                continue
            question = str(sample.get("question", "")).strip()
            if question and is_chinese_text(question):
                prompt = "请详细解答下面的数学问题，给出清晰的推导步骤和最终答案：\\n\\n" + question
                if check_prompt_length(prompt):
                    prompts["math-zh"].append(prompt)
                    if len(prompts["math-zh"]) >= N_PREFETCH:
                        break
        if len(prompts["math-zh"]) >= N_PREFETCH:
            break


    # Supplement Chinese mathematics with native Chinese word problems
    # from CMATH, then translated GSM8K questions if more candidates are needed.
    math_zh_seen = set(prompts["math-zh"])

    def add_math_zh_question(question):
        question = str(question or "").strip()
        if not question or not is_chinese_text(question):
            return

        prompt = (
            "请详细解答下面的数学问题，给出清晰的推导步骤和最终答案：\n\n"
            + question
        )

        if prompt in math_zh_seen or not check_prompt_length(prompt):
            return

        prompts["math-zh"].append(prompt)
        math_zh_seen.add(prompt)

    # 1. Native Chinese math word problems.
    if len(prompts["math-zh"]) < N_PREFETCH:
        ds_name = "weitianwen/cmath"
        print(f"Loading {ds_name}...")
        cmath = datasets.load_dataset(ds_name, split="validation")

        for sample in cmath.shuffle(seed=shuffle_seed):
            add_math_zh_question(sample.get("question", ""))

            if len(prompts["math-zh"]) >= N_PREFETCH:
                break

    # 2. Additional Chinese-translated grade-school math problems.
    if len(prompts["math-zh"]) < N_PREFETCH:
        ds_name = "meta-math/GSM8K_zh"
        print(f"Loading {ds_name}...")
        gsm8k_zh = datasets.load_dataset(ds_name, split="train")

        for sample in gsm8k_zh.shuffle(seed=shuffle_seed):
            add_math_zh_question(sample.get("question_zh", ""))

            if len(prompts["math-zh"]) >= N_PREFETCH:
                break

    print(f"math-zh candidate prompts after supplementation: "
          f"{len(prompts['math-zh'])}")


    # Chinese coding instructions. The "code" configuration contains prompt/response pairs.
    ds_name = "Mxode/Chinese-Instruct-Lite"
    print(f"Loading {ds_name} code...")
    dataset = datasets.load_dataset(ds_name, "code", split="train")
    for sample in tqdm.tqdm(dataset.shuffle(seed=shuffle_seed), desc=ds_name + " code"):
        q = str(sample.get("prompt", "")).strip()
        a = str(sample.get("response", "")).strip()
        if (
            q
            and a
            and is_chinese_text(q)
            and check_prompt_length(q)
        ):
            prompts["code-zh"].append(q)
            if len(prompts["code-zh"]) >= N_PREFETCH:
                break

    # MERA-evaluation/MERA
    ds_name = "MERA-evaluation/MERA"
    print(f"Loading {ds_name}...")
    configs = datasets.get_dataset_config_names(ds_name)
    n = 0
    for config in configs:
        for split_name in ["test", "validation"]:
            try:
                splits = datasets.get_dataset_split_names(ds_name, config)
                if split_name not in splits:
                    continue

                dataset = datasets.load_dataset(ds_name, config, split=split_name)
                for sample in tqdm.tqdm(dataset, desc=ds_name + ":" + config + ":" + split_name):
                    instruct = sample['instruction']
                    if isinstance(sample['inputs'], str):
                        instruct = re.sub(r'\{.+?\}', sample['inputs'], instruct)
                    else:
                        for key, value in sample['inputs'].items():
                            instruct = instruct.replace('{' + key + '}', str(value))

                    if config == 'multiq':
                        output = sample['outputs']
                    elif config == "ruhumaneval":
                        if 'function' in sample['inputs']:
                            output = sample['inputs']['function'] + sample['meta']["canonical_solution"]
                        else:
                            output = sample['meta']["canonical_solution"]
                    elif isinstance(sample['outputs'], str):
                        output = sample['outputs'].strip()
                    else:
                        raise NotImplementedError()

                    q = instruct
                    a = output
                    if check_prompt_length(q) and a:
                        if config in ["ruhumaneval", "rucodeeval"]:
                            prompts["code-ru"].append(q)
                        else:
                            prompts["qa-ru"].append(q)

            except Exception as ex:
                print(traceback.format_exc())

    if len(prompts["code-ru"]) < N_PREFETCH:
        with open("coding_data/ru_coding_instruct_dataset.jsonl") as f:
            for line in f:
                sample = json.loads(line)
                prompt = sample["query"]
                if check_prompt_length(prompt) and is_russian_text(prompt[:100]):
                    prompts["code-ru"].append(prompt)

    ds_name = "wmt/wmt19"
    print(f"Loading {ds_name}...")
    dataset = datasets.load_dataset(ds_name, "ru-en", split="validation")

    ru_en_pairs = []
    for example in dataset:
        src = example["translation"]["ru"]
        ref = example["translation"]["en"]
        # отбираем тексты не слишком короткие (в wmt полно однословных сэмплов).
        if check_prompt_length(src) and 200 < len(ref) < 2000 and is_russian_text(src):
            prompts["wmt ru-en"].append(f"Translate the text into English: {src}")

    ds_name = "me-aas/python-code-dataset-500k"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="train")
    for sample in ds.shuffle(seed=shuffle_seed).take(N_PREFETCH):
        prompts["code"].append(sample["instruction"])

    ds_name = "HaimingW/math_train_decontaminated"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="train")
    for sample in ds.shuffle(seed=shuffle_seed).take(N_PREFETCH):
        prompts["math"].append(sample["problem"])

    ds_name = "checkai/instruction-poems"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="train")
    for sample in ds.shuffle(seed=shuffle_seed).take(N_PREFETCH):
        prompts["poetry-en"].append(sample["INSTRUCTION"])

    ds_name = "IsDeeCee/StoryMaker"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="train")
    for sample in ds.shuffle(seed=shuffle_seed):
        prompts["creative-en"].append(sample["Prompt"])

    ds_name = "smd20/social-engineering-qa-english"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="train")
    for sample in ds.shuffle(seed=shuffle_seed).take(N_PREFETCH):
        prompts["qa"].append(sample["question"])

    # ---------------------------------------------------------------------
    # Let’s make a final selection of prompts that will elicit the longest
    # possible responses from the model (batched greedy generation).
    final_prompts = collections.defaultdict(list)

    # Per-domain properties of prompts and corresponding responses
    qa_prompts = collections.defaultdict(list)
    math_prompts = collections.defaultdict(list)
    code_prompts = collections.defaultdict(list)
    poetry_prompts = collections.defaultdict(list)

    print(f"Loading {model_id}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",  # options: sdpa | eager | flash_attention_2 | flash_attention_4
        trust_remote_code=True,
    ).eval()

    info_table = [["Domain", "No. prompts", "Prompt length, toks<br>mean<sub>min, max</sub>", "Response length, toks<br>mean<sub>min, max</sub>"]]
    for idomain, domain in enumerate(sorted(prompts.keys()), start=1):
        # НАЧАЛО ОТЛАДКИ
        #if domain != "poetry-zh":
        #    continue
        # КОНЕЦ ОТЛАДКИ

        print("\n{}/{} Start processing domain \"{}\"".format(idomain, len(prompts), domain))

        raw_prompts = prompts[domain]
        raw_prompts = list(dict.fromkeys(raw_prompts))  # avoid duplicates; keeps the order, so the selection is reproducible
        if domain.endswith("-zh"):
            raw_prompts = [prompt for prompt in raw_prompts if check_chinese_prompt_length(prompt)]
        else:
            raw_prompts = [prompt for prompt in raw_prompts if check_prompt_length(prompt)]
        raw_prompts = sorted(raw_prompts, key=lambda z: -len(z))
        print("Raw prompts: {}".format(len(raw_prompts)))

        if not raw_prompts:
            print(f"WARNING: no candidate prompts collected for domain {domain}; skipping.")
            continue

        if len(raw_prompts) > N_REPEATS*4:
            raw_prompts = raw_prompts[:N_REPEATS*4]
            print("No. raw prompts after truncation: {}".format(len(raw_prompts)))

        prompt_response_list = measure_response_lengths(model, tokenizer, raw_prompts, GEN_BATCH_SIZE, MAX_NEW_TOKENS,
                                                        desc=domain)

        prompt_response_list = sorted(prompt_response_list, key=lambda z: -z[2])  # stable: ties keep the original order
        prompt_response_list = prompt_response_list[:N_REPEATS]

        if domain.endswith("-ru"):
            language = "ru"
        elif domain.endswith("-zh"):
            language = "zh"
        else:
            language = "en"

        for prompt, prompt_num_toks, response_num_toks in prompt_response_list:
            final_prompts[domain].append(prompt)
            prompt_info = {"prompt": prompt,
                           "prompt_num_toks": prompt_num_toks,
                           "response_num_toks": response_num_toks}
            if domain.startswith("math"):
                math_prompts[language].append(prompt_info)
            elif domain.startswith("qa"):
                qa_prompts[language].append(prompt_info)
            elif domain.startswith("code"):
                code_prompts[language].append(prompt_info)
            elif domain.startswith("poetry"):
                poetry_prompts[language].append(prompt_info)

        prompt_lengths = [prompt_len for _, prompt_len, _ in prompt_response_list]
        response_lengths = [response_len for _, _, response_len in prompt_response_list]
        if not response_lengths:
            print(f"WARNING: no measured prompts for domain {domain}; skipping statistics.")
            final_prompts.pop(domain, None)
            continue

        print("Domain: {}  population: {}  response lengths: min={} avg={} max={}".format(domain, len(final_prompts[domain]), min(response_lengths), np.mean(response_lengths), max(response_lengths)))

        info_table.append((domain,
                           len(final_prompts[domain]),
                           "{:.1f}<sub>{}, {}</sub>".format(np.mean(prompt_lengths), min(prompt_lengths), max(prompt_lengths)),
                           "{:.1f}<sub>{}, {}</sub>".format(np.mean(response_lengths), min(response_lengths), max(response_lengths))
                           ))

        # After processing each domain, we output the selected prompts to a file for visual review.
        with open("speed_evaluation_dataset.txt", "w") as f:
            for domain, domain_prompts in final_prompts.items():
                f.write(f"\n=== START OF DOMAIN: {domain} ===\n\n")
                for i, prompt in enumerate(domain_prompts, start=1):
                    f.write(f"--- domain: {domain}  prompt #{i} ---\n\n{prompt}\n\n")
                f.write(f"\n=== END OF DOMAIN {domain} ===\n\n")

        # Visualize prompt and response lengths for several domains
        for domain, stat in [("code", code_prompts), ("qa", qa_prompts), ("math", math_prompts), ("poetry", poetry_prompts)]:
            if stat:
                language_to_prompt_lengths = dict()
                for language, stat in stat.items():
                    language_to_prompt_lengths[language] = [x["response_num_toks"] for x in stat]

                plt.clf()
                fig, zx = plot_grouped_length_histograms(domain, language_to_prompt_lengths, bins=5, density=True)
                fig.savefig(f"response_length_in_domain_{domain}.png", dpi=150, bbox_inches="tight")

        plt.close()

    # Generation speed evaluation dataset in JSON file.
    with open("speed_evaluation_dataset.json", "w") as f:
        json.dump(final_prompts, f, indent=4, ensure_ascii=False)

    # Dataset statistics in Markdown format.
    with open("speed_evaluation_dataset.md", "w") as f:
        f.write("Tokenizer: `{}`\n\n".format(model_id))
        f.write(terminaltables.GithubFlavoredMarkdownTable(info_table).table)
