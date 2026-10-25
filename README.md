# Language Model Generation Speed Evaluation Dataset

A multilingual prompt dataset for benchmarking text-generation
throughput across language models and inference methods, including
speculative decoding.

## Overview

The dataset contains **1,200 prompts across 12 domains**, with **100
prompts per domain**. The domains cover English- and Russian-language
tasks, including mathematics, code generation, grammar correction,
question answering, poetry, creative writing, and Russian-to-English
translation.

Prompts were selected with the aim of eliciting sufficiently long
responses from **Qwen3-0.6B**, so that throughput measurements are less
dominated by short generations. The dataset can be used to compare
ordinary autoregressive decoding with acceleration methods such as
speculative decoding and lossless drafter-acceptance schemes.

The dataset measures generation performance, not task quality. It does
not provide reference answers or define a quality-scoring protocol.

## Files

-   [`speed_evaluation_dataset.json`](speed_evaluation_dataset.json) ---
    the prompts, grouped by domain.
-   [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) ---
    prompt and response length statistics for each domain, measured with
    the `Qwen/Qwen3-0.6B` tokenizer.
-   [`build_speed_evaluation_dataset.py`](build_speed_evaluation_dataset.py)
    --- the script used to construct the dataset.

## Domains

  Domain key      Task
  --------------- -------------------------------------------------------
  `poetry-ru`     Russian poetry generation  
  `gec-ru`        Russian grammar, punctuation, and spelling correction  
  `gec-en`        English grammar, punctuation, and spelling correction  
  `math-ru`       Russian-language mathematics problems  
  `qa-ru`         Russian-language question answering and instructions  
  `code-ru`       Russian-language coding tasks  
  `wmt ru-en`     Translation from Russian into English  
  `code`          English-language coding tasks  
  `math`          English-language mathematics problems  
  `poetry-en`     English poetry generation  
  `creative-en`   English creative writing  
  `qa`            English-language question answering  

There are six Russian-oriented and six English-oriented domain
categories. Each category contains 100 prompts.

## Dataset construction

The construction script collects candidate prompts from a combination of
public datasets and private collections. Public sources used by the
script include:

-   `jhu-clsp/jfleg` --- English grammatical error correction
-   `evilfreelancer/MATH-500-Russian` --- Russian mathematics
-   `agentlans/allenai-WildChat-4.8M` --- English and Russian
    question-answering prompts
-   `MERA-evaluation/MERA` --- Russian-language tasks, including coding
-   `wmt/wmt19` --- Russian-to-English translation
-   `me-aas/python-code-dataset-500k` --- coding
-   `HaimingW/math_train_decontaminated` --- mathematics
-   `checkai/instruction-poems` --- English poetry
-   `IsDeeCee/StoryMaker` --- creative writing
-   `smd20/social-engineering-qa-english` --- question answering

The script also uses the Russian grammar-correction resources RUPOR and
LoRuGEC, plus private prompt collections for Russian poetry. Some of
these sources are local files and are not included in this repository.

Candidate prompts are deduplicated within each domain and filtered for
basic length constraints. The script uses the `Qwen/Qwen3-0.6B` model to
generate greedy responses for candidate prompts, then selects up to 100
prompts per domain by descending generated response length. The
response-length probe allows at most 2,048 new tokens per prompt.
Consequently, some recorded response lengths reach this cap; these are
capped measurements, not necessarily the model's natural stopping
lengths.

The construction script uses a fixed shuffle seed for applicable dataset
sampling. However, reproducing the exact dataset also depends on
source-dataset versions, model and library versions, and access to the
private/local sources.

## Using the dataset

The JSON file is a mapping from domain names to lists of prompt strings.
A minimal loading example:

``` python
import json
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/Koziev/SpeedEvaluationDataset/main/speed_evaluation_dataset.json"

with urlopen(url) as response:
    prompts_by_domain = json.load(response)

for domain, prompts in prompts_by_domain.items():
    print(domain, len(prompts))
    for prompt in prompts:
        # Run the prompt through the model or inference server being evaluated.
        pass
```

For throughput benchmarking, run the same prompts with the same
evaluation settings for each model or decoding method. Report the
complete measurement setup, including at least:

-   model and tokenizer versions;
-   hardware and inference framework;
-   precision and serving configuration;
-   batch size and concurrency;
-   generation limits and stopping criteria;
-   whether prompt processing, queueing, and other serving overheads are
    included.

Measure throughput as generated output tokens divided by elapsed
wall-clock time, using a consistent token-counting convention across
systems. To compare end-to-end generation speed, include the overheads
that occur in the intended deployment path rather than timing only the
model's token-generation kernel. For speculative decoding, also report
relevant method-specific settings, such as the drafter configuration and
acceptance statistics, where available.

The prompts were selected using Qwen3-0.6B response lengths, so the set
is not guaranteed to produce equally long outputs from other models.
Some generations may terminate early, and some prompts may hit a
generation limit. Interpret aggregate throughput together with
per-domain results and output-length statistics.

## Length statistics

See [`speed_evaluation_dataset.md`](speed_evaluation_dataset.md) for the
per-domain mean, minimum, and maximum prompt and response lengths
measured with the `Qwen/Qwen3-0.6B` tokenizer. These statistics describe
the selection run and are not expected to match token counts from other
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

This dataset is intended for research and engineering comparisons of
language-model inference throughput, including evaluations of
speculative decoding and related acceleration techniques. For meaningful
comparisons, keep the evaluation protocol consistent and publish the
settings needed to reproduce the measurements.
