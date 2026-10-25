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
from transformers import AutoModelForCausalLM, AutoTokenizer
import unicodedata


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


@torch.inference_mode()
def measure_response_lengths(model, tokenizer, prompts, batch_size: int, max_new_tokens: int, desc: str = ""):
    """
    Greedy response length (tokens up to and including the first stop token, at most max_new_tokens) for every prompt,
    generated in left-padded batches. Prompts are batched in order of token length, so a batch carries little padding;
    a batch that runs out of GPU memory is split in half and retried. Returns [(prompt, prompt_len, response_len)] in the
    order of `prompts`.
    """
    stop_ids = stop_token_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(stop_ids)
    encoded = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                             enable_thinking=False, return_dict=True)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -len(encoded[i]))     # longest first: an OOM shows up early
    response_len = [0] * len(prompts)

    def run(idx):
        L = max(len(encoded[i]) for i in idx)
        input_ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids = encoded[i]
            input_ids[r, L - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[r, L - len(ids):] = 1
        try:
            out = model.generate(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                                 do_sample=False, temperature=None, top_p=None, top_k=None,
                                 max_new_tokens=max_new_tokens, pad_token_id=pad_id)
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(idx) // 2
            run(idx[:half])
            run(idx[half:])
            return
        gen = out[:, L:].tolist()
        for r, i in enumerate(idx):
            n = next((k + 1 for k, t in enumerate(gen[r]) if t in stop_ids), len(gen[r]))
            response_len[i] = min(n, max_new_tokens)

    batches = [order[k:k + batch_size] for k in range(0, len(order), batch_size)]
    for idx in tqdm.tqdm(batches, desc=desc):
        run(idx)
    return [(p, len(e), n) for p, e, n in zip(prompts, encoded, response_len)]


if __name__ == "__main__":
    N_REPEATS = 100             # Number of prompts to keep per domain
    N_PREFETCH = 2 * N_REPEATS  # Number of prompts to sample from domain pool before ranking and selecting.
    GEN_BATCH_SIZE = 32         # prompts per generate() call; an OOM batch is split automatically
    MAX_NEW_TOKENS = 2048       # response-length probe; responses reaching it count as MAX_NEW_TOKENS
    MAX_PROMPT_TOKENS = MAX_NEW_TOKENS//2

    prompts = collections.defaultdict(list)

    shuffle_seed = 31415926

    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_id}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",  # options: sdpa | eager | flash_attention_2 | flash_attention_4
        trust_remote_code=True,
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")

    # ---------------------------------------------------------------------------

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

    ds_name = "evilfreelancer/MATH-500-Russian"
    print(f"Loading {ds_name}...")
    ds = datasets.load_dataset(ds_name, split="test")
    for sample in ds.shuffle(seed=shuffle_seed):
        prompts["math-ru"].append(sample["problem"])

    ds_name = "agentlans/allenai-WildChat-4.8M"
    dataset = datasets.load_dataset(ds_name, "Russian", split="train", streaming=False)
    for sample in tqdm.tqdm(dataset, desc=ds_name+" Russian"):
        messages = sample["conversations"]
        if len(messages) == 2 and messages[0]["from"] == "human" and messages[1]["from"] == "gpt":
            q = messages[0]["value"].strip()
            a = messages[1]["value"].strip()
            if 200 < len(q) and 200 < len(a) and is_russian_text(q) and is_russian_text(a):
                prompts["qa-ru"].append(q)
                if len(prompts["qa-ru"]) >= N_PREFETCH:
                    break

    dataset = datasets.load_dataset(ds_name, "English", split="train", streaming=False)
    for sample in tqdm.tqdm(dataset, desc=ds_name+" English"):
        messages = sample["conversations"]
        if len(messages) == 2 and messages[0]["from"] == "human" and messages[1]["from"] == "gpt":
            q = messages[0]["value"].strip()
            a = messages[1]["value"].strip()
            if 200 < len(q) and 200 < len(a):
                prompts["qa"].append(q)
                if len(prompts["qa"]) >= N_PREFETCH:
                    break

    # Chinese general question answering.
    ds_name = "agentlans/allenai-WildChat-4.8M"
    print(f"Loading {ds_name} Chinese...")
    dataset = datasets.load_dataset(ds_name, "Chinese", split="train", streaming=False)
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
            ):
                prompts["qa-zh"].append(q)
                if len(prompts["qa-zh"]) >= N_PREFETCH:
                    break

    # Chinese mathematics. LiveK12Bench provides Chinese and English splits;
    # use only Chinese splits and retain math questions.
    ds_name = "Shawn-wxh/livek12bench"
    print(f"Loading {ds_name} Chinese mathematics...")
    livek12 = datasets.load_dataset(ds_name)
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

    # Chinese poetry-generation instructions. Filter a broad Chinese instruction
    # corpus for prompts specifically about composing poetry or classical verse.
    ds_name = "lcccher/Chinese-Instruct"
    print(f"Loading {ds_name} poetry candidates...")
    dataset = datasets.load_dataset(ds_name, "firefly", split="train")
    poetry_terms = ("写一首诗", "写首诗", "创作一首诗", "创作诗歌", "写一首古诗",
                    "作一首诗", "写诗", "诗歌创作", "填词", "写一首词", "对联")
    for sample in tqdm.tqdm(dataset.shuffle(seed=shuffle_seed), desc=ds_name + " poetry"):
        q = str(sample.get("prompt", "")).strip()
        a = str(sample.get("response", "")).strip()
        if (
            q
            and a
            and is_chinese_text(q)
            and is_chinese_text(a)
            and any(term in q for term in poetry_terms)
            and check_prompt_length(q)
        ):
            prompts["poetry-zh"].append(q)
            if len(prompts["poetry-zh"]) >= N_PREFETCH:
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
    for sample in ds.shuffle(seed=shuffle_seed).take(N_PREFETCH):
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
    info_table = [["Domain", "No. prompts", "Mean prompt length, toks", "Mean response length, toks"]]
    for domain, raw_prompts in prompts.items():

        # DEBUG BEGIN
        #if domain != "qa-ru":
        #    continue
        # DEBUG END

        raw_prompts = list(dict.fromkeys(raw_prompts))  # avoid duplicates; keeps the order, so the selection is reproducible
        print("\nDomain: {}  raw prompts: {}".format(domain, len(raw_prompts)))

        if len(raw_prompts) > N_PREFETCH:
            raw_prompts = [prompt for prompt in tqdm.tqdm(raw_prompts, desc=domain) if check_prompt_length(prompt)]
            raw_prompts = sorted(raw_prompts, key=lambda z: -len(z))[:N_PREFETCH]

        if not raw_prompts:
            print(f"WARNING: no candidate prompts collected for domain {domain}; skipping.")
            continue

        prompt_response_list = measure_response_lengths(model, tokenizer, raw_prompts, GEN_BATCH_SIZE, MAX_NEW_TOKENS,
                                                        desc=domain)

        prompt_response_list = sorted(prompt_response_list, key=lambda z: -z[2])  # stable: ties keep the original order
        prompt_response_list = prompt_response_list[:N_REPEATS]
        for prompt, _, _ in prompt_response_list:
            final_prompts[domain].append(prompt)

        prompt_lengths = [prompt_len for _, prompt_len, _ in prompt_response_list]
        response_lengths = [response_len for _, _, response_len in prompt_response_list]
        if not response_lengths:
            print(f"WARNING: no measured prompts for domain {domain}; skipping statistics.")
            final_prompts.pop(domain, None)
            continue

        print("Domain: {}  response lengths: min={} avg={} max={}".format(domain, min(response_lengths), np.mean(response_lengths), max(response_lengths)))

        info_table.append((domain,
                           len(final_prompts[domain]),
                           "{:.1f}<sub>{}, {}</sub>".format(np.mean(prompt_lengths), min(prompt_lengths), max(prompt_lengths)),
                           "{:.1f}<sub>{}, {}</sub>".format(np.mean(response_lengths), min(response_lengths), max(response_lengths))
                           ))

    # Generation speed evaluation dataset in JSON file.
    with open("speed_evaluation_dataset.json", "w") as f:
        json.dump(final_prompts, f, indent=4, ensure_ascii=False)

    # Dataset statistics in Markdown format.
    with open("speed_evaluation_dataset.md", "w") as f:
        f.write("Tokenizer: `{}`\n\n".format(model_id))
        f.write(terminaltables.GithubFlavoredMarkdownTable(info_table).table)

    # Output the lists of prompts for visual control.
    with open("speed_evaluation_dataset.txt", "w") as f:
        for domain, prompts in final_prompts.items():
            f.write(f"\n=== START OF DOMAIN: {domain} ===\n\n")
            for i, prompt in enumerate(prompts, start=1):
                f.write(f"--- domain {domain} prompt #{i} ---\n\n{prompt}\n\n")
            f.write(f"\n=== END OF DOMAIN {domain} ===\n\n")
