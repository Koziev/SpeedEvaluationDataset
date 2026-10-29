# Language Model Generation Speed Evaluation Datasets

I maintain these datasets to compare language-model inference speed across ordinary text generation, agentic coding, and tool calling.
I've used them in my research on speculative decoding and alternative LLM architectures (see, for example, [arXiv:2609.15504](https://arxiv.org/abs/2609.15504) and [arXiv:2609.32814](https://arxiv.org/abs/2609.32814)), and I hope they'll be useful to other researchers as well.

I provide two datasets with different formats:

- The **first dataset** contains standalone text prompts grouped by domain.
- The **second dataset** contains structured conversation trajectories, including prior messages and, for tool-calling examples, tool definitions.

These datasets are designed to measure generation speed, not task quality. They do not provide a general quality-scoring protocol.

## First dataset: multilingual prompts

### Overview

The first dataset contains up to **1,600 prompts across 16 domains**, covering English, Russian, and Chinese. The tasks include mathematics, code generation, grammar correction, question answering, poetry, creative writing, and Russian-to-English translation.

I selected prompts to encourage [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to generate relatively long responses, so throughput measurements are less dominated by short outputs. This selection strategy does not guarantee equally long responses from other models.

### Files

- [`speed_evaluation_dataset.json`](speed_evaluation_dataset.json) — prompts grouped by domain.
- [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) — prompt and response length statistics for each domain, measured with the `Qwen/Qwen3-0.6B` tokenizer.
- [`build_speed_evaluation_dataset_v2.py`](build_speed_evaluation_dataset_v2.py) — the script I use to build the first dataset.

### Domains

| Domain key | Task |
|---|---|
| `qa` | English-language question answering |
| `qa-ru` | Russian-language question answering and instructions |
| `qa-zh` | Chinese-language general question answering |
| `math` | English-language mathematics problems |
| `math-ru` | Russian-language mathematics problems |
| `math-zh` | Chinese-language mathematics problems |
| `code` | English-language coding tasks |
| `code-ru` | Russian-language coding tasks |
| `code-zh` | Chinese-language coding tasks |
| `poetry-en` | English poetry generation |
| `poetry-ru` | Russian poetry generation |
| `poetry-zh` | Chinese poetry generation |
| `gec-en` | English grammar, punctuation, and spelling correction |
| `gec-ru` | Russian grammar, punctuation, and spelling correction |
| `wmt ru-en` | Translation from Russian into English |
| `creative-en` | English creative writing |

I aim for 100 prompts per domain. A domain may contain fewer if not enough candidates pass filtering and response-length selection.

### Sources

I build the first dataset from public datasets and private collections. The sources include:

- [`jhu-clsp/jfleg`](https://huggingface.co/datasets/jhu-clsp/jfleg) — English grammatical error correction.
- [`evilfreelancer/MATH-500-Russian`](https://huggingface.co/datasets/evilfreelancer/MATH-500-Russian) — Russian mathematics.
- [`agentlans/allenai-WildChat-4.8M`](https://huggingface.co/datasets/agentlans/allenai-WildChat-4.8M) — English, Russian, and Chinese question-answering prompts.
- [`MERA-evaluation/MERA`](https://huggingface.co/datasets/MERA-evaluation/MERA) — Russian-language tasks, including coding.
- [`Koziev/Translations`](https://github.com/Koziev/Translations) — Russian-English translation pairs.
- [`wmt/wmt19`](https://huggingface.co/datasets/wmt/wmt19) — Russian-to-English translation.
- [`me-aas/python-code-dataset-500k`](https://huggingface.co/datasets/me-aas/python-code-dataset-500k) — coding instructions.
- [`HaimingW/math_train_decontaminated`](https://huggingface.co/datasets/HaimingW/math_train_decontaminated) — mathematics.
- [`checkai/instruction-poems`](https://huggingface.co/datasets/checkai/instruction-poems) — English poetry.
- [`IsDeeCee/StoryMaker`](https://huggingface.co/datasets/IsDeeCee/StoryMaker) — creative writing.
- [`smd20/social-engineering-qa-english`](https://huggingface.co/datasets/smd20/social-engineering-qa-english) — question answering.
- **RUPOR** ([arXiv:2505.04507](https://arxiv.org/abs/2505.04507)) and **LoRuGEC** ([paper](https://aclanthology.org/2025.bea-1.38/)) — Russian grammar-correction resources.
- Private prompt collections for Russian poetry.

Some Russian sources are local files and are not included in this repository.

For the Chinese domains, I use these sources and selection rules:

| Domain | Source | Selection approach |
|---|---|---|
| `qa-zh` | [`agentlans/allenai-WildChat-4.8M`](https://huggingface.co/datasets/agentlans/allenai-WildChat-4.8M), `Chinese` configuration | I select user prompts from two-message human/assistant conversations when both the query and response exceed 200 characters and pass the Chinese-text filter. |
| `math-zh` | [`Shawn-wxh/livek12bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench) | I select Chinese records from the `zh_2603` and `zh_2605` splits whose subject is mathematics, then wrap each question in an instruction asking for a detailed solution. I randomly paraphrase this instruction prefix to introduce additional prompt variation. |
| `code-zh` | [`Mxode/Chinese-Instruct-Lite`](https://huggingface.co/datasets/Mxode/Chinese-Instruct-Lite), `code` configuration | I sample Chinese coding instructions with non-empty prompt and response fields. |
| `poetry-zh` | [`lcccher/Chinese-Instruct`](https://huggingface.co/datasets/lcccher/Chinese-Instruct), `firefly` configuration | I filter Chinese instructions for poetry-related terms, including requests for poems, classical verse, lyrics, and couplets. I randomly paraphrase the poetry instruction prefix to introduce additional prompt variation. |

My Chinese-text filter checks whether Han characters account for more than half of the relevant characters after whitespace and punctuation are excluded. This is a practical heuristic in the construction script, not a comprehensive language-identification system.

Dataset availability, schemas, and licensing terms can change. Please check the source dataset cards and licenses before redistributing derived data or using it commercially. In particular, [`LiveK12Bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench) is marked CC BY-NC 4.0, which limits commercial use.

### How I build the first dataset

I deduplicate candidate prompts within each domain and filter them using basic length constraints. I then use `Qwen/Qwen3-0.6B` to generate greedy responses and select up to 100 prompts per domain, prioritizing prompts with longer generated responses.

The response-length probe allows at most 2,048 new tokens per prompt. Some recorded response lengths therefore reach this cap; those measurements do not necessarily represent the model's natural stopping lengths.

I use a fixed shuffle seed where applicable. Reproducing the exact dataset also depends on source-dataset versions, model and library versions, and access to private or local sources. A domain may contain fewer than 100 prompts if a source does not provide enough qualifying candidates.

### Loading the first dataset

The JSON file maps domain names to lists of prompt strings. This example downloads it from GitHub using only Python's standard library:

```python
import json
from urllib.request import urlopen

url = (
    "https://raw.githubusercontent.com/"
    "Koziev/SpeedEvaluationDataset/main/"
    "speed_evaluation_dataset.json"
)

with urlopen(url) as response:
    prompts_by_domain = json.load(response)

for domain, prompts in prompts_by_domain.items():
    print(domain, len(prompts))
    for prompt in prompts:
        # Run the prompt through the model or inference server you're evaluating.
        pass
```

## Second dataset: agentic coding and tool calling

I added a second dataset for benchmarking inference on longer conversational contexts and structured assistant tool calls. By default, it contains **100 samples in each of two domains** (200 samples total).

### Files

- [`speed_evaluation_dataset_part2.json`](speed_evaluation_dataset_part2.json) — conversation trajectories for agentic coding and tool calling.
- [`speed_evaluation_dataset_part2.md`](speed_evaluation_dataset_part2.md) — per-sample and aggregate token-length statistics.
- [`build_speed_evaluation_dataset_part2.py`](build_speed_evaluation_dataset_part2.py) — the script I use to build the dataset and generate its Markdown statistics report.
- [`smoke_test_speed_evaluation_dataset_part2.py`](smoke_test_speed_evaluation_dataset_part2.py) — a smoke test that runs examples from the second dataset with Qwen models.

### Domains

| Domain key | Contents | Source |
|---|---|---|
| `agentic_coding` | Multi-message agentic software-engineering trajectories, with an assistant turn selected as the prediction target | [`thoughtworks/agentic-coding-trajectories`](https://huggingface.co/datasets/thoughtworks/agentic-coding-trajectories) |
| `tool_calling` | Function/tool-calling conversations, including declared tool schemas and assistant tool-call targets | [`Archangel-system/glaive-function-calling-v2-openai-native`](https://huggingface.co/datasets/Archangel-system/glaive-function-calling-v2-openai-native) |

### Sampling and structure

- **Agentic coding:** I select valid trajectories across the source dataset's `max_isl` length bands to retain a range of context sizes. The source `max_isl` values are measured with `cl100k_base`; they're included as metadata and should not be treated as Qwen token counts.
- **Tool calling:** I filter the source records to those categorized as `tool_call`, then select examples with a usable assistant tool-call target and tool definitions.
- The output is a JSON **list of sample objects**, unlike the first dataset's JSON object mapping domain names to lists of plain prompt strings.
- Each sample contains a `domain` and a normalized `messages` conversation. Agentic-coding samples store the target turn index in `metadata.target_message_index`; tool-calling samples store it in `validation.target_message_index`. Tool-calling samples also include `tools` and validation metadata, including expected tool calls and function names.
- The target index identifies the assistant message to predict. For inference, pass only messages **before** that index as input; don't include the target message in the prompt. For tool-calling examples, include the provided tool definitions when your inference interface supports them.

The Markdown report records model-specific input and target token lengths using each selected tokenizer's chat template. Input lengths include the serialized conversation context and, for tool-calling examples, tool definitions. Expected target lengths include the serialized target assistant turn. These are reference lengths, not generated output lengths or measures of task quality.

### Building the second dataset

The builder requires the Hugging Face `datasets` and `transformers` packages, as well as access to the source datasets and model tokenizers. From the repository directory, run:

```bash
pip install -U datasets transformers
python build_speed_evaluation_dataset_part2.py
```

By default, the script writes `speed_evaluation_dataset_part2.json` and `speed_evaluation_dataset_part2.md`, with 100 examples per domain. It accepts `--samples-per-domain`, `--seed`, `--agentic-split`, `--tool-split`, `--stats-models`, and `--output`; run `python build_speed_evaluation_dataset_part2.py --help` for details. The default statistics tokenizers are `Qwen/Qwen3-0.6B` and `Qwen/Qwen3.5-0.8B`.

### Loading the second dataset

```python
import json

with open("speed_evaluation_dataset_part2.json", encoding="utf-8") as f:
    samples = json.load(f)

for sample in samples:
    domain = sample["domain"]
    messages = sample["messages"]

    if domain == "agentic_coding":
        target_index = sample["metadata"]["target_message_index"]
        tools = None
    elif domain == "tool_calling":
        target_index = sample["validation"]["target_message_index"]
        tools = sample["tools"]
    else:
        continue

    prompt_messages = messages[:target_index]
    expected_target = messages[target_index]

    # Serialize prompt_messages with the evaluated model's chat template.
    # For tool_calling, also pass `tools` when the serving API supports them.
```

The `smoke_test_speed_evaluation_dataset_part2.py` script provides a small end-to-end check with Qwen models. It verifies that samples can be serialized and run; it is not a quality evaluation.

## Benchmarking guidance

For a fair throughput comparison, I recommend running the same examples with the same evaluation settings for every model or decoding method. The two datasets have different input formats:

- For the first dataset, each item is a standalone prompt.
- For the second dataset, preserve the conversation history and tool definitions, and use the target-message index to build the input without leaking the expected target.

Report the full measurement setup, including:

- model and tokenizer versions;
- hardware and inference framework;
- precision and serving configuration;
- batch size and concurrency;
- generation limits and stopping criteria;
- whether prompt processing, queueing, and other serving overheads are included.

Measure throughput as generated output tokens divided by elapsed wall-clock time, using a consistent token-counting convention across systems. If you want to compare end-to-end generation speed, include the overheads that occur in your intended deployment path rather than timing only the model's token-generation kernel. For speculative decoding, also report relevant method-specific settings, such as the drafter configuration and acceptance statistics, when available.

The first dataset was selected using Qwen3-0.6B response lengths, so other models may produce shorter or longer outputs. Some generations may terminate early or hit a generation limit. The second dataset's target turns come from recorded trajectories; treat their lengths as reference values, not as outputs that the evaluated model must reproduce exactly. I recommend interpreting aggregate throughput alongside per-domain results and output-length statistics.

## Length statistics

- [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) contains per-domain mean, minimum, and maximum prompt and response lengths for the first dataset, measured with the `Qwen/Qwen3-0.6B` tokenizer.
- [`speed_evaluation_dataset_part2.md`](speed_evaluation_dataset_part2.md) contains input and target-turn token-length summaries for the second dataset, measured with the listed model tokenizers and chat templates.

Token counts depend on the tokenizer and chat template, so they may differ from counts produced by other tokenizers or serving configurations.

## Limitations

- **Response-length selection bias:** I selected prompts for long greedy responses from Qwen3-0.6B, so the first dataset is not a random sample of typical user traffic.
- **Model dependence:** prompts that elicit long responses from Qwen3-0.6B may behave differently with other models.
- **Capped length probes:** response-length measurements stop at 2,048 new tokens, so values at the cap are limited by the probe rather than necessarily reflecting the model's natural stopping point.
- **Source and licensing considerations:** the datasets draw from multiple public sources and private collections. Please check the relevant licenses, terms of use, and redistribution requirements for your use case.
- **Not a quality benchmark:** these datasets measure inference speed; they do not, by themselves, measure correctness, usefulness, or safety.
- **Different data formats:** the first dataset contains plain prompt strings grouped by domain. The second contains structured message trajectories; tool-calling examples also include tool schemas and expected tool-call metadata. Your benchmark harness needs to handle these formats correctly.

## Intended use

I built these datasets to support research and engineering comparisons of language-model inference throughput, including speculative decoding and related acceleration techniques. For meaningful comparisons, keep the evaluation protocol consistent and publish enough details for others to reproduce your measurements.
