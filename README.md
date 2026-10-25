# Language Model Generation Speed Evaluation Dataset

A multilingual prompt dataset for benchmarking text-generation throughput across language models and inference methods, including
speculative decoding.

## Overview

The dataset is designed to contain **100 prompts in each of 16 domains** (up to **1,600 prompts** in total), covering English, Russian, and
Chinese. Tasks include mathematics, code generation, grammar correction, question answering, poetry, creative writing, and Russian-to-English
translation.

Prompts are selected with the aim of eliciting sufficiently long responses from **Qwen3-0.6B**, so that throughput measurements are less
dominated by short generations. The dataset can be used to compare ordinary autoregressive decoding with acceleration methods such as
speculative decoding and lossless drafter-acceptance schemes.

The dataset measures generation performance, not task quality. It does not provide reference answers or define a quality-scoring protocol.

## Files

-   [`speed_evaluation_dataset.json`](speed_evaluation_dataset.json) ---
    prompts grouped by domain.
-   [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) ---
    prompt and response length statistics for each domain, measured with
    the `Qwen/Qwen3-0.6B` tokenizer.
-   [`build_speed_evaluation_dataset_v2.py`](build_speed_evaluation_dataset_v2.py)
    --- the script used to construct the dataset.

## Domains

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

The domain names identify the task and language. The dataset-building script targets 100 prompts per domain; the actual number available in a
generated dataset depends on how many candidates pass filtering and response-length selection.

## Dataset sources

The construction script collects candidate prompts from public datasets and private collections. Sources include:

### English and Russian domains

-   [`jhu-clsp/jfleg`](https://huggingface.co/datasets/jhu-clsp/jfleg)
    --- English grammatical error correction.
-   [`evilfreelancer/MATH-500-Russian`](https://huggingface.co/datasets/evilfreelancer/MATH-500-Russian)
    --- Russian mathematics.
-   [`agentlans/allenai-WildChat-4.8M`](https://huggingface.co/datasets/agentlans/allenai-WildChat-4.8M)
    --- English, Russian, and Chinese question-answering prompts.
-   [`MERA-evaluation/MERA`](https://huggingface.co/datasets/MERA-evaluation/MERA)
    --- Russian-language tasks, including coding.
-   [`wmt/wmt19`](https://huggingface.co/datasets/wmt/wmt19) ---
    Russian-to-English translation.
-   [`me-aas/python-code-dataset-500k`](https://huggingface.co/datasets/me-aas/python-code-dataset-500k)
    --- coding instructions.
-   [`HaimingW/math_train_decontaminated`](https://huggingface.co/datasets/HaimingW/math_train_decontaminated)
    --- mathematics.
-   [`checkai/instruction-poems`](https://huggingface.co/datasets/checkai/instruction-poems)
    --- English poetry.
-   [`IsDeeCee/StoryMaker`](https://huggingface.co/datasets/IsDeeCee/StoryMaker)
    --- creative writing.
-   [`smd20/social-engineering-qa-english`](https://huggingface.co/datasets/smd20/social-engineering-qa-english)
    --- question answering.
-   **RUPOR and LoRuGEC** --- Russian grammar-correction resources.
-   Private prompt collections for Russian poetry.

Some Russian sources are local files and are not included in this repository.

### Chinese domains

| Domain | Source                                                                                                                        | Selection approach                                                                                                                                                                         |
|---|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`qa-zh`| [`agentlans/allenai-WildChat-4.8M`](https://huggingface.co/datasets/agentlans/allenai-WildChat-4.8M), `Chinese` configuration | Select user prompts<br>from two-message<br>human/assistant<br>conversations when both<br>the query and response<br>exceed 200 characters<br>and pass the<br>Chinese-text filter.           |
|`math-zh`| [`Shawn-wxh/livek12bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench) | Select Chinese records<br>in the `zh_2603` and<br>`zh_2605` splits whose<br>subject is mathematics.<br>Each question is<br>wrapped in an<br>instruction asking for<br>a detailed solution. |
|`code-zh`| [`Mxode/Chinese-Instruct-Lite`](https://huggingface.co/datasets/Mxode/Chinese-Instruct-Lite), `code` configuration | Sample Chinese coding<br>instructions with<br>non-empty prompt and<br>response fields. |
|`poetry-zh`| [`lcccher/Chinese-Instruct`](https://huggingface.co/datasets/lcccher/Chinese-Instruct), `firefly` configuration | Filter Chinese<br>instructions for<br>poetry-related terms,<br>including requests for<br>poems, classical verse,<br>lyrics, and couplets.|
                                                                                                                                 
The Chinese text filter checks whether Han characters account for more than half of the relevant characters after whitespace and punctuation
are excluded. It is a practical script-level heuristic, not a comprehensive language-identification system.

Dataset availability, schemas, and licensing terms may change. Review the source dataset cards and licenses before redistributing derived data
or using it in commercial settings. In particular, [`LiveK12Bench`](https://huggingface.co/datasets/Shawn-wxh/livek12bench)
is marked CC BY-NC 4.0, which limits commercial use.

## Dataset construction

Candidate prompts are deduplicated within each domain and filtered for basic length constraints. The script uses the `Qwen/Qwen3-0.6B` model to
generate greedy responses for candidate prompts, then selects up to 100 prompts per domain by descending generated response length. The
response-length probe allows at most 2,048 new tokens per prompt. Consequently, some recorded response lengths reach this cap; these are
capped measurements, not necessarily the model's natural stopping lengths.

The script uses a fixed shuffle seed for applicable dataset sampling. Reproducing the exact dataset also depends on source-dataset versions,
model and library versions, and access to private or local sources. If a source does not yield enough qualifying candidates, a domain may contain
fewer than 100 prompts.

## Using the dataset

The JSON file maps domain names to lists of prompt strings. The following example downloads the dataset directly from GitHub using only
Python's standard library:

``` python
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
        # Run the prompt through the model or inference server being evaluated.
        pass
```

For throughput benchmarking, run the same prompts with the same evaluation settings for each model or decoding method. Report the
complete measurement setup, including at least:

-   model and tokenizer versions;
-   hardware and inference framework;
-   precision and serving configuration;
-   batch size and concurrency;
-   generation limits and stopping criteria;
-   whether prompt processing, queueing, and other serving overheads are
    included.

Measure throughput as generated output tokens divided by elapsed wall-clock time, using a consistent token-counting convention across
systems. To compare end-to-end generation speed, include the overheads that occur in the intended deployment path rather than timing only the
model's token-generation kernel. For speculative decoding, also report relevant method-specific settings, such as the drafter configuration and
acceptance statistics, where available.

The prompts were selected using Qwen3-0.6B response lengths, so the set is not guaranteed to produce equally long outputs from other models.
Some generations may terminate early, and some prompts may hit a generation limit. Interpret aggregate throughput together with
per-domain results and output-length statistics.

## Length statistics

See [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) for per-domain mean, minimum, and maximum prompt and response lengths
measured with the `Qwen/Qwen3-0.6B` tokenizer. These statistics describe the selection run and are not expected to match token counts from other
tokenizers or generation configurations.

## Limitations

-   **Response-length selection bias:** prompts were selected for long
    greedy responses from Qwen3-0.6B. The dataset is therefore not a
    random sample of typical user traffic.
-   **Model dependence:** prompts that elicit long responses from
    Qwen3-0.6B may behave differently with other models.
-   **Capped length probes:** response-length measurements stop at 2,048
    new tokens, so values at the cap are right-censored by the probe
    limit.
-   **Source and licensing considerations:** prompts originate from
    multiple datasets and private collections. Users are responsible for
    checking the applicable licenses, terms of use, and redistribution
    requirements for their intended use.
-   **Not a quality benchmark:** the dataset is intended for speed
    evaluation; it does not by itself measure correctness, usefulness,
    or safety.

## Intended use

This dataset is intended for research and engineering comparisons of language-model inference throughput, including evaluations of
speculative decoding and related acceleration techniques. For meaningful comparisons, keep the evaluation protocol consistent and publish the
settings needed to reproduce the measurements.
