# Speed Evaluation Dataset — Part 2: Dataset Statistics

This report describes `speed_evaluation_dataset_part2.json`, containing agentic-coding trajectories and tool-calling examples.

## Dataset composition

| Domain | Samples | Source dataset |
|---|---:|---|
| `agentic_coding` | 100 | `thoughtworks/agentic-coding-trajectories` |
| `tool_calling` | 100 | `Archangel-system/glaive-function-calling-v2-openai-native` |

## Token-length summary

Lengths are computed using each model's own tokenizer and chat template. `Input tokens` count the serialized conversation prefix through the target turn boundary, with the assistant generation prompt and (for tool calling) the tool definitions included. `Expected target tokens` are estimated as the token-count difference between that prefix and the serialized prefix plus the reference assistant target turn. This includes template control tokens and tool-call serialization, so it is more useful than tokenizing the message's plain-text `content` alone. Small boundary effects from tokenization/template serialization are possible.

| Domain | Tokenizer/model | N | Input min | Input median | Input mean | Input p90 | Input max | Target min | Target median | Target mean | Target p90 | Target max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `agentic_coding` | `Qwen/Qwen3-0.6B` | 100 | 4,742 | 25,804 | 32,400 | 70,024 | 113,236 | 11 | 140 | 235 | 677 | 1,158 |
| `agentic_coding` | `Qwen/Qwen3.5-0.8B` | 100 | 4,922 | 27,104 | 33,602 | 71,911 | 114,651 | 7 | 138 | 230 | 651 | 1,164 |
| `tool_calling` | `Qwen/Qwen3-0.6B` | 100 | 165 | 265 | 275 | 361 | 465 | 22 | 44 | 45 | 59 | 88 |
| `tool_calling` | `Qwen/Qwen3.5-0.8B` | 100 | 288 | 388 | 398 | 484 | 588 | 16 | 55 | 54 | 75 | 105 |

## Per-sample token lengths

One row per sample. `Input` and `target` lengths are tokenizer-specific. For agentic-coding samples, `Source max ISL` is the source dataset's `max_isl` value measured with `cl100k_base`; it is included for reference and is not directly comparable to Qwen token counts. ISL stands for *Input Sequence Length*.

### `agentic_coding`

| Sample ID | Messages | Prompt messages | Source max ISL (cl100k) | Qwen/Qwen3-0.6B input | Qwen/Qwen3-0.6B target | Qwen/Qwen3.5-0.8B input | Qwen/Qwen3.5-0.8B target |
|---|---:|---:|---:|---:|---:|---:|---:|
| `kwai-klear-swe-smith-mini::luozhouyang__python-string-similarity.115acaac.lm_rewrite__f9p7evxv` | 35 | 34 | 7,459 | 7,895 | 152 | 8,201 | 150 |
| `swe-smith-claude-3-7-sonnet::getmoto__moto.694ce1f4.func_pm_remove_wrapper__7470mf9d.91hmylr4` | 75 | 74 | 22,302 | 25,373 | 17 | 26,379 | 14 |
| `nebius-swe-rebench-openhands::chatcmpl-c09ddd183c1a6b49d4cb9c6e06ec8d87` | 183 | 182 | 69,474 | 69,237 | 894 | 70,961 | 900 |
| `swe-smith-claude-3-7-sonnet::john-kurkowski__tldextract.3d1bf184.combine_file__a8cw58y5.cb713a4o` | 83 | 82 | 43,599 | 46,356 | 17 | 48,696 | 14 |
| `nebius-swe-rebench-openhands::chatcmpl-2f2f51c728590bc57f94f1a16d96053b` | 201 | 200 | 68,394 | 67,401 | 48 | 69,687 | 51 |
| `nebius-swe-rebench-openhands::chatcmpl-f51c15b16b64a61b91cc8d321306efc0` | 93 | 92 | 34,954 | 37,610 | 517 | 38,865 | 518 |
| `nebius-swe-rebench-openhands::chatcmpl-67bcf206c65921a1abd23224bf5d5c4a` | 95 | 94 | 30,159 | 29,713 | 615 | 30,787 | 590 |
| `swe-smith-claude-3-7-sonnet::pytest-dev__iniconfig.16793ead.combine_module__sxsvwi72.3qdud4nz` | 27 | 26 | 6,071 | 6,624 | 11 | 7,001 | 7 |
| `kwai-klear-swe-smith-mini::PyCQA__flake8.cf1542ce.lm_rewrite__g5gdve70` | 119 | 118 | 24,335 | 25,667 | 111 | 26,898 | 108 |
| `nebius-swe-rebench-openhands::chatcmpl-52ea8285b4e5d74bf9745e95d991be55` | 75 | 74 | 30,025 | 30,553 | 447 | 31,830 | 421 |
| `swe-smith-claude-3-7-sonnet::sqlfluff__sqlfluff.50a1c4b6.func_pm_ctrl_invert_if__5yzxd7rw.91hmylr4` | 27 | 26 | 5,487 | 5,846 | 11 | 6,091 | 7 |
| `kwai-klear-swe-smith-mini::mahmoud__boltons.3bfcfdd0.combine_file__zrhbqatu` | 41 | 40 | 9,808 | 10,249 | 203 | 10,741 | 202 |
| `kwai-klear-swe-smith-mini::cantools__cantools.0c6a7871.lm_rewrite__87z8vxa8` | 69 | 68 | 14,350 | 15,250 | 148 | 15,836 | 145 |
| `kwai-klear-swe-smith-mini::oauthlib__oauthlib.1fd52536.lm_rewrite__1mk7jsct` | 101 | 100 | 21,001 | 22,209 | 179 | 23,207 | 180 |
| `kwai-klear-swe-smith-mini::bottlepy__bottle.a8dfef30.func_basic__mzx9j8vd` | 21 | 20 | 4,407 | 4,742 | 124 | 4,922 | 122 |
| `kwai-klear-swe-smith-mini::pyupio__safety.7654596b.combine_module__6x1lxddx` | 23 | 22 | 5,966 | 6,116 | 139 | 6,477 | 136 |
| `swe-smith-claude-3-7-sonnet::pallets__click.fde47b4b.combine_file__0p8nh9y7.cb713a4o` | 153 | 152 | 25,415 | 29,436 | 15 | 30,567 | 11 |
| `swe-smith-claude-3-7-sonnet::getmoto__moto.694ce1f4.pr_4950.kv86f030` | 47 | 46 | 16,000 | 17,308 | 36 | 18,092 | 33 |
| `nebius-swe-rebench-openhands::chatcmpl-fdf527ee32a4f6990b389a6703b33eed` | 183 | 182 | 70,600 | 70,024 | 529 | 71,902 | 503 |
| `kwai-klear-swe-smith-mini::joke2k__faker.8b401a7d.combine_module__lggd16za` | 27 | 26 | 7,342 | 7,844 | 184 | 8,232 | 184 |
| `nebius-swe-rebench-openhands::chatcmpl-149409c4881809e60e069731451274e2` | 97 | 96 | 44,231 | 46,329 | 655 | 47,900 | 628 |
| `kwai-klear-swe-smith-mini::Suor__funcy.207a7810.func_pm_op_change__vt0hxx5a` | 151 | 150 | 36,407 | 38,140 | 66 | 39,872 | 62 |
| `swe-smith-claude-3-7-sonnet::pydicom__pydicom.7d361b3d.func_pm_ctrl_invert_if__phni80ab.91hmylr4` | 49 | 48 | 19,419 | 22,076 | 11 | 22,895 | 7 |
| `kwai-klear-swe-smith-mini::pyupio__safety.7654596b.combine_module__av5fdunh` | 51 | 50 | 13,283 | 13,684 | 310 | 14,481 | 308 |
| `kwai-klear-swe-smith-mini::paramiko__paramiko.23f92003.lm_rewrite__jgg9soic` | 63 | 62 | 16,304 | 17,519 | 116 | 18,404 | 112 |
| `kwai-klear-swe-smith-mini::agronholm__typeguard.b6a7e438.combine_module__m6txo517` | 29 | 28 | 7,258 | 7,561 | 191 | 7,961 | 184 |
| `swe-smith-claude-3-7-sonnet::pyutils__line_profiler.a646bf0f.lm_rewrite__72bwnkym.3qdud4nz` | 77 | 76 | 35,630 | 38,279 | 18 | 40,132 | 15 |
| `nebius-swe-rebench-openhands::chatcmpl-f1b2911cba14a094e7279d27aa564648` | 123 | 122 | 66,182 | 66,710 | 838 | 68,262 | 836 |
| `nebius-swe-rebench-openhands::chatcmpl-df73b941d2ef262c08471080c7129fbd` | 135 | 134 | 69,709 | 75,150 | 442 | 77,469 | 418 |
| `kwai-klear-swe-smith-mini::luozhouyang__python-string-similarity.115acaac.lm_rewrite__iphd5rii` | 31 | 30 | 7,045 | 7,427 | 236 | 7,721 | 237 |
| `swe-smith-claude-3-7-sonnet::graphql-python__graphene.82903263.lm_rewrite__gx4fxdu3.3qdud4nz` | 51 | 50 | 16,155 | 17,753 | 11 | 18,489 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-db4436ab441297229b3bf574247a18d2` | 201 | 200 | 91,647 | 94,111 | 254 | 97,736 | 260 |
| `kwai-klear-swe-smith-mini::Cog-Creators__Red-DiscordBot.33e0eac7.combine_module__kh7gk9le` | 41 | 40 | 10,196 | 10,660 | 142 | 11,368 | 139 |
| `kwai-klear-swe-smith-mini::PyCQA__flake8.cf1542ce.lm_rewrite__lxo1k2je` | 31 | 30 | 5,105 | 5,464 | 108 | 5,665 | 102 |
| `nebius-swe-rebench-openhands::chatcmpl-c0717fd6e8da6d4e696471241dc4a47c` | 85 | 84 | 51,720 | 51,918 | 1,054 | 53,319 | 1,044 |
| `swe-smith-claude-3-7-sonnet::joke2k__faker.8b401a7d.func_pm_remove_wrapper__psvo3kzw.91hmylr4` | 31 | 30 | 22,505 | 23,878 | 17 | 24,411 | 14 |
| `kwai-klear-swe-smith-mini::pyasn1__pyasn1.0f07d724.combine_file__c2630w6g` | 23 | 22 | 5,185 | 5,358 | 129 | 5,643 | 128 |
| `kwai-klear-swe-smith-mini::pyupio__safety.7654596b.combine_module__o6v39izp` | 99 | 98 | 23,370 | 24,310 | 68 | 25,750 | 64 |
| `kwai-klear-swe-smith-mini::rsalmei__alive-progress.35853799.pr_241` | 151 | 150 | 27,825 | 29,740 | 311 | 30,955 | 310 |
| `kwai-klear-swe-smith-mini::pygments__pygments.27649ebb.pr_2624` | 27 | 26 | 7,466 | 8,321 | 162 | 8,667 | 160 |
| `swe-smith-claude-3-7-sonnet::joke2k__faker.8b401a7d.func_pm_class_rm_funcs__zy4u4n2x.91hmylr4` | 49 | 48 | 24,813 | 26,703 | 11 | 27,536 | 7 |
| `kwai-klear-swe-smith-mini::google__textfsm.c31b6007.combine_file__wfv4qh8m` | 37 | 36 | 11,400 | 11,859 | 214 | 12,690 | 211 |
| `kwai-klear-swe-smith-mini::bottlepy__bottle.a8dfef30.func_pm_remove_assign__s8d7sc7k` | 41 | 40 | 7,809 | 8,274 | 197 | 8,642 | 197 |
| `swe-smith-claude-3-7-sonnet::getnikola__nikola.0f4c230e.lm_rewrite__d7sbe6yx.3qdud4nz` | 85 | 84 | 31,464 | 34,010 | 17 | 35,746 | 14 |
| `nebius-swe-rebench-openhands::chatcmpl-9b94ab3d9edeedda755b1816e9c72757` | 111 | 110 | 29,047 | 28,277 | 449 | 29,595 | 442 |
| `swe-smith-claude-3-7-sonnet::pygments__pygments.27649ebb.lm_rewrite__0kem80sh.3qdud4nz` | 43 | 42 | 33,426 | 34,822 | 17 | 36,318 | 14 |
| `kwai-klear-swe-smith-mini::joke2k__faker.8b401a7d.lm_rewrite__y7ndnmr8` | 37 | 36 | 11,541 | 12,279 | 186 | 12,953 | 183 |
| `swe-smith-claude-3-7-sonnet::getnikola__nikola.0f4c230e.func_pm_ctrl_shuffle__qyvdy0r6.91hmylr4` | 91 | 90 | 28,830 | 30,690 | 11 | 32,086 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-1b66b4c926b5c5c27981d2b123cdc573` | 143 | 142 | 67,724 | 69,469 | 1,158 | 72,287 | 1,164 |
| `nebius-swe-rebench-openhands::chatcmpl-b977f75fd3862275c1fc7dc84154ab26` | 137 | 136 | 51,132 | 51,254 | 841 | 52,922 | 835 |
| `swe-smith-claude-3-7-sonnet::davidhalter__parso.338a5760.lm_rewrite__ynfi8c6q.3qdud4nz` | 23 | 22 | 7,450 | 8,206 | 17 | 8,509 | 14 |
| `kwai-klear-swe-smith-mini::python-openxml__python-docx.0cf6d71f.combine_module__g6ptzfwa` | 73 | 72 | 29,075 | 30,160 | 69 | 31,744 | 65 |
| `swe-smith-claude-3-7-sonnet::pandas-dev__pandas.95280573.func_pm_ctrl_invert_if__kvgcfa67.91hmylr4` | 153 | 152 | 31,746 | 36,624 | 15 | 38,040 | 11 |
| `kwai-klear-swe-smith-mini::pdfminer__pdfminer.six.1a8bd2f7.func_basic__sajz1lzn` | 51 | 50 | 14,760 | 15,496 | 234 | 16,459 | 234 |
| `kwai-klear-swe-smith-mini::cantools__cantools.0c6a7871.combine_file__c733290x` | 103 | 102 | 24,376 | 25,941 | 165 | 27,505 | 164 |
| `nebius-swe-rebench-openhands::chatcmpl-6626fefa3fd4f3ef4ce508344d11d2ca` | 103 | 102 | 47,886 | 47,110 | 687 | 48,529 | 664 |
| `kwai-klear-swe-smith-mini::agronholm__typeguard.b6a7e438.func_basic__03o4fz9t` | 39 | 38 | 8,566 | 8,928 | 89 | 9,442 | 85 |
| `swe-smith-claude-3-7-sonnet::andialbrecht__sqlparse.e57923b3.lm_rewrite__2ldlhnt2.3qdud4nz` | 137 | 136 | 66,033 | 72,010 | 15 | 75,738 | 11 |
| `kwai-klear-swe-smith-mini::seatgeek__thefuzz.8a05a3ee.func_basic__3yvg2tul` | 39 | 38 | 10,234 | 10,799 | 160 | 11,271 | 158 |
| `kwai-klear-swe-smith-mini::paramiko__paramiko.23f92003.combine_file__w0fox91v` | 27 | 26 | 7,151 | 7,765 | 151 | 8,089 | 148 |
| `nebius-swe-rebench-openhands::chatcmpl-ed848d495e2641d24cb539ea9e4097fb` | 115 | 114 | 60,427 | 60,800 | 521 | 62,985 | 499 |
| `swe-smith-claude-3-7-sonnet::gruns__icecream.f76fef56.func_pm_ctrl_invert_if__sy2hagpa.3qdud4nz` | 25 | 24 | 8,297 | 9,226 | 11 | 9,628 | 7 |
| `swe-smith-claude-3-7-sonnet::pallets__click.fde47b4b.lm_rewrite__ytci3086.3qdud4nz` | 153 | 152 | 26,817 | 30,305 | 15 | 31,531 | 11 |
| `swe-smith-claude-3-7-sonnet::mahmoud__glom.fb3c4e76.lm_rewrite__b6dzct3q.3qdud4nz` | 67 | 66 | 17,001 | 18,919 | 11 | 19,657 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-473f5fb0460c7990ae3dbd5adafe4b5e` | 169 | 168 | 64,809 | 63,061 | 895 | 64,628 | 904 |
| `swe-smith-claude-3-7-sonnet::pandas-dev__pandas.95280573.pr_57764.4tw4cp8h` | 121 | 120 | 39,166 | 42,416 | 11 | 44,269 | 7 |
| `kwai-klear-swe-smith-mini::marshmallow-code__webargs.dbde72fe.combine_module__8qynwkf7` | 49 | 48 | 13,299 | 13,703 | 180 | 14,361 | 174 |
| `swe-smith-claude-3-7-sonnet::python-hyper__h11.bed0dd4a.func_pm_remove_cond__h7z75jl4.3qdud4nz` | 55 | 54 | 23,307 | 26,258 | 17 | 27,311 | 14 |
| `nebius-swe-rebench-openhands::chatcmpl-c04c12273ee47c2f7a8581a15eb6465c` | 201 | 200 | 65,033 | 65,975 | 85 | 68,432 | 89 |
| `swe-smith-claude-3-7-sonnet::cantools__cantools.0c6a7871.combine_file__96u51ak6.cb713a4o` | 93 | 92 | 31,738 | 35,361 | 11 | 36,876 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-395d3e6e5029a1ef896be54f65911e37` | 183 | 182 | 82,953 | 83,138 | 694 | 86,611 | 652 |
| `kwai-klear-swe-smith-mini::pyasn1__pyasn1.0f07d724.combine_module__zztpjfaq` | 31 | 30 | 6,705 | 6,950 | 115 | 7,366 | 114 |
| `kwai-klear-swe-smith-mini::pyupio__safety.7654596b.combine_file__vtudncut` | 53 | 52 | 20,869 | 21,766 | 72 | 22,825 | 68 |
| `swe-smith-claude-3-7-sonnet::facebookresearch__hydra.0f03eb60.pr_1551.kv86f030` | 45 | 44 | 11,377 | 12,839 | 59 | 13,271 | 56 |
| `kwai-klear-swe-smith-mini::PyCQA__flake8.cf1542ce.lm_rewrite__tdi0h139` | 61 | 60 | 16,851 | 17,693 | 123 | 18,711 | 120 |
| `nebius-swe-rebench-openhands::chatcmpl-a255fe75f9175488e81a289ea7fcea7b` | 91 | 90 | 51,585 | 52,823 | 688 | 54,065 | 672 |
| `kwai-klear-swe-smith-mini::cantools__cantools.0c6a7871.lm_rewrite__qkzu91gj` | 65 | 64 | 18,174 | 20,226 | 177 | 20,755 | 175 |
| `kwai-klear-swe-smith-mini::luozhouyang__python-string-similarity.115acaac.lm_rewrite__f9p7evxv` | 29 | 28 | 6,252 | 6,660 | 139 | 6,909 | 138 |
| `kwai-klear-swe-smith-mini::PyCQA__flake8.cf1542ce.lm_rewrite__nzd9ovhl` | 105 | 104 | 21,690 | 22,535 | 170 | 23,822 | 166 |
| `swe-smith-claude-3-7-sonnet::pydata__patsy.a5d16484.func_pm_ctrl_invert_if__id1j34qt.91hmylr4` | 153 | 152 | 69,835 | 74,304 | 15 | 77,052 | 11 |
| `kwai-klear-swe-smith-mini::pyupio__safety.7654596b.combine_module__b4m47adc` | 105 | 104 | 25,204 | 26,655 | 94 | 27,969 | 91 |
| `swe-smith-claude-3-7-sonnet::Project-MONAI__MONAI.a09c1f08.pr_5627.3qdud4nz` | 39 | 38 | 29,528 | 30,588 | 11 | 31,140 | 7 |
| `kwai-klear-swe-smith-mini::Suor__funcy.207a7810.lm_rewrite__dyvutlua` | 145 | 144 | 46,924 | 49,110 | 243 | 51,293 | 240 |
| `swe-smith-claude-3-7-sonnet::paramiko__paramiko.23f92003.func_pm_remove_assign__7x073tlp.91hmylr4` | 55 | 54 | 19,554 | 21,085 | 11 | 22,100 | 7 |
| `kwai-klear-swe-smith-mini::oauthlib__oauthlib.1fd52536.lm_rewrite__pf6xh62j` | 71 | 70 | 13,423 | 14,040 | 189 | 14,750 | 188 |
| `swe-smith-claude-3-7-sonnet::pandas-dev__pandas.95280573.pr_51741.oosvj10h` | 55 | 54 | 102,771 | 113,236 | 17 | 114,651 | 13 |
| `nebius-swe-rebench-openhands::chatcmpl-06516c48d0ded6246b84f796669f795b` | 191 | 190 | 82,687 | 82,793 | 432 | 85,111 | 408 |
| `swe-smith-claude-3-7-sonnet::theskumar__python-dotenv.2b8635b7.combine_module__6ec1qz1l.3qdud4nz` | 43 | 42 | 13,456 | 14,694 | 17 | 15,336 | 14 |
| `swe-smith-claude-3-7-sonnet::cantools__cantools.0c6a7871.combine_file__tfklw0we.3qdud4nz` | 61 | 60 | 23,202 | 25,175 | 19 | 26,298 | 16 |
| `kwai-klear-swe-smith-mini::pygments__pygments.27649ebb.lm_rewrite__3tqlni0u` | 33 | 32 | 12,480 | 12,937 | 243 | 13,473 | 243 |
| `swe-smith-claude-3-7-sonnet::django-money__django-money.835c1ab8.combine_module__bqggrnha.3qdud4nz` | 47 | 46 | 17,151 | 18,876 | 11 | 19,738 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-fbc473672cda2e1e943f314994470043` | 175 | 174 | 72,172 | 70,407 | 657 | 72,025 | 632 |
| `nebius-swe-rebench-openhands::chatcmpl-823c9de5c1336ada23e40006d34ae61a` | 133 | 132 | 56,116 | 56,684 | 574 | 58,318 | 556 |
| `nebius-swe-rebench-openhands::chatcmpl-b5c611a76324a65bd3178b4767d7f9a4` | 175 | 174 | 73,393 | 74,819 | 988 | 77,093 | 984 |
| `kwai-klear-swe-smith-mini::oauthlib__oauthlib.1fd52536.func_basic__jfzqihyy` | 45 | 44 | 10,260 | 10,946 | 165 | 11,491 | 162 |
| `swe-smith-claude-3-7-sonnet::facelessuser__soupsieve.a8080d97.func_pm_ctrl_shuffle__wbgs5zj3.3qdud4nz` | 133 | 132 | 31,300 | 34,916 | 11 | 36,494 | 7 |
| `nebius-swe-rebench-openhands::chatcmpl-db365d26bea71563c80f9a52801fbb6e` | 135 | 134 | 69,364 | 70,106 | 379 | 71,911 | 371 |
| `nebius-swe-rebench-openhands::chatcmpl-4d64ce03ab9cbcbb45982e6ce53ede30` | 131 | 130 | 57,245 | 58,080 | 446 | 59,915 | 421 |
| `kwai-klear-swe-smith-mini::pdfminer__pdfminer.six.1a8bd2f7.func_basic__zctj4eba` | 23 | 22 | 5,279 | 5,774 | 102 | 6,003 | 98 |
| `nebius-swe-rebench-openhands::chatcmpl-adfbd23242fb0eb542b0fcaf56d4c60f` | 131 | 130 | 54,842 | 53,618 | 677 | 55,464 | 651 |

### `tool_calling`

| Sample ID | Messages | Prompt messages | Source max ISL (cl100k) | Qwen/Qwen3-0.6B input | Qwen/Qwen3-0.6B target | Qwen/Qwen3.5-0.8B input | Qwen/Qwen3.5-0.8B target |
|---|---:|---:|---:|---:|---:|---:|---:|
| `row-1455` | 7 | 4 | — | 320 | 45 | 443 | 60 |
| `row-1096` | 9 | 4 | — | 299 | 48 | 422 | 61 |
| `row-1064` | 5 | 2 | — | 244 | 36 | 367 | 45 |
| `row-49` | 11 | 6 | — | 260 | 22 | 383 | 16 |
| `row-1225` | 9 | 2 | — | 289 | 33 | 412 | 36 |
| `row-487` | 9 | 2 | — | 205 | 43 | 328 | 46 |
| `row-1190` | 9 | 4 | — | 242 | 26 | 365 | 27 |
| `row-30` | 5 | 2 | — | 299 | 34 | 422 | 38 |
| `row-1187` | 7 | 4 | — | 285 | 68 | 408 | 72 |
| `row-1618` | 5 | 2 | — | 316 | 49 | 440 | 64 |
| `row-1483` | 9 | 4 | — | 333 | 63 | 455 | 80 |
| `row-435` | 11 | 6 | — | 375 | 46 | 498 | 59 |
| `row-1758` | 5 | 2 | — | 315 | 88 | 438 | 105 |
| `row-1260` | 5 | 2 | — | 232 | 42 | 355 | 52 |
| `row-404` | 9 | 2 | — | 362 | 34 | 485 | 41 |
| `row-377` | 5 | 2 | — | 374 | 71 | 497 | 80 |
| `row-96` | 11 | 2 | — | 299 | 27 | 422 | 29 |
| `row-1736` | 11 | 2 | — | 243 | 42 | 366 | 57 |
| `row-528` | 7 | 4 | — | 340 | 52 | 463 | 82 |
| `row-1299` | 9 | 4 | — | 274 | 41 | 397 | 48 |
| `row-1595` | 5 | 2 | — | 197 | 29 | 320 | 32 |
| `row-410` | 9 | 2 | — | 257 | 36 | 380 | 45 |
| `row-1335` | 11 | 2 | — | 283 | 28 | 406 | 30 |
| `row-216` | 5 | 2 | — | 252 | 36 | 375 | 44 |
| `row-665` | 5 | 2 | — | 261 | 51 | 384 | 63 |
| `row-1182` | 7 | 2 | — | 229 | 45 | 352 | 55 |
| `row-1009` | 7 | 2 | — | 222 | 52 | 345 | 62 |
| `row-853` | 7 | 4 | — | 287 | 48 | 410 | 60 |
| `row-1001` | 7 | 2 | — | 193 | 38 | 316 | 41 |
| `row-349` | 11 | 2 | — | 243 | 37 | 366 | 46 |
| `row-703` | 9 | 4 | — | 435 | 48 | 558 | 61 |
| `row-775` | 11 | 2 | — | 272 | 33 | 395 | 42 |
| `row-913` | 7 | 4 | — | 465 | 58 | 588 | 75 |
| `row-1593` | 7 | 2 | — | 237 | 52 | 360 | 62 |
| `row-114` | 9 | 4 | — | 432 | 55 | 555 | 66 |
| `row-513` | 7 | 4 | — | 394 | 53 | 517 | 68 |
| `row-813` | 7 | 4 | — | 352 | 80 | 475 | 85 |
| `row-1427` | 7 | 4 | — | 281 | 32 | 404 | 40 |
| `row-648` | 5 | 2 | — | 264 | 57 | 387 | 71 |
| `row-690` | 7 | 2 | — | 222 | 54 | 345 | 63 |
| `row-883` | 7 | 2 | — | 201 | 36 | 324 | 39 |
| `row-907` | 9 | 2 | — | 370 | 49 | 493 | 64 |
| `row-1171` | 5 | 2 | — | 340 | 45 | 463 | 54 |
| `row-1235` | 11 | 2 | — | 165 | 22 | 288 | 17 |
| `row-1391` | 9 | 4 | — | 231 | 26 | 354 | 28 |
| `row-262` | 13 | 4 | — | 336 | 58 | 459 | 68 |
| `row-1684` | 7 | 2 | — | 194 | 39 | 317 | 43 |
| `row-1506` | 7 | 4 | — | 298 | 54 | 421 | 62 |
| `row-667` | 11 | 2 | — | 195 | 27 | 318 | 29 |
| `row-1311` | 9 | 4 | — | 306 | 38 | 429 | 51 |
| `row-1379` | 7 | 4 | — | 360 | 58 | 483 | 75 |
| `row-709` | 7 | 2 | — | 302 | 85 | 425 | 98 |
| `row-1138` | 9 | 2 | — | 205 | 28 | 328 | 30 |
| `row-259` | 7 | 2 | — | 232 | 32 | 355 | 40 |
| `row-807` | 11 | 2 | — | 224 | 36 | 347 | 45 |
| `row-1420` | 5 | 2 | — | 351 | 61 | 474 | 79 |
| `row-1417` | 7 | 2 | — | 220 | 50 | 343 | 58 |
| `row-296` | 5 | 2 | — | 323 | 37 | 446 | 46 |
| `row-1169` | 7 | 2 | — | 238 | 50 | 361 | 60 |
| `row-1128` | 11 | 2 | — | 184 | 27 | 307 | 30 |
| `row-983` | 3 | 2 | — | 223 | 54 | 345 | 62 |
| `row-1381` | 7 | 2 | — | 250 | 59 | 373 | 75 |
| `row-448` | 7 | 2 | — | 221 | 36 | 344 | 45 |
| `row-149` | 9 | 6 | — | 360 | 40 | 483 | 48 |
| `row-804` | 7 | 4 | — | 367 | 48 | 490 | 61 |
| `row-1606` | 7 | 2 | — | 227 | 50 | 350 | 60 |
| `row-1014` | 7 | 2 | — | 185 | 37 | 308 | 40 |
| `row-504` | 7 | 2 | — | 271 | 57 | 394 | 67 |
| `row-1123` | 7 | 4 | — | 291 | 45 | 414 | 59 |
| `row-761` | 9 | 2 | — | 232 | 35 | 355 | 38 |
| `row-1470` | 7 | 4 | — | 311 | 44 | 434 | 59 |
| `row-583` | 9 | 4 | — | 288 | 46 | 410 | 58 |
| `row-157` | 7 | 2 | — | 208 | 51 | 331 | 56 |
| `row-1555` | 7 | 2 | — | 230 | 54 | 353 | 63 |
| `row-1705` | 7 | 4 | — | 246 | 35 | 369 | 37 |
| `row-525` | 7 | 2 | — | 205 | 50 | 328 | 55 |
| `row-226` | 7 | 4 | — | 319 | 46 | 442 | 61 |
| `row-560` | 7 | 2 | — | 192 | 27 | 315 | 29 |
| `row-1` | 9 | 6 | — | 261 | 22 | 384 | 17 |
| `row-1733` | 7 | 2 | — | 267 | 59 | 390 | 75 |
| `row-1118` | 5 | 2 | — | 319 | 36 | 442 | 45 |
| `row-1043` | 7 | 2 | — | 188 | 36 | 311 | 40 |
| `row-1218` | 5 | 2 | — | 346 | 37 | 469 | 46 |
| `row-1111` | 5 | 2 | — | 218 | 52 | 341 | 55 |
| `row-898` | 7 | 4 | — | 361 | 84 | 484 | 98 |
| `row-808` | 7 | 2 | — | 236 | 53 | 359 | 63 |
| `row-1456` | 9 | 2 | — | 222 | 31 | 345 | 39 |
| `row-1042` | 9 | 2 | — | 283 | 33 | 406 | 41 |
| `row-1623` | 7 | 2 | — | 267 | 72 | 390 | 87 |
| `row-84` | 11 | 2 | — | 233 | 27 | 356 | 29 |
| `row-76` | 7 | 2 | — | 225 | 52 | 348 | 62 |
| `row-1328` | 9 | 2 | — | 369 | 49 | 492 | 58 |
| `row-107` | 9 | 4 | — | 280 | 41 | 403 | 51 |
| `row-217` | 7 | 2 | — | 259 | 65 | 382 | 83 |
| `row-1428` | 9 | 4 | — | 230 | 29 | 353 | 30 |
| `row-1643` | 3 | 2 | — | 308 | 38 | 431 | 46 |
| `row-801` | 13 | 4 | — | 252 | 36 | 375 | 45 |
| `row-684` | 7 | 4 | — | 318 | 50 | 441 | 65 |
| `row-955` | 3 | 2 | — | 222 | 39 | 344 | 46 |
| `row-505` | 7 | 4 | — | 274 | 35 | 397 | 43 |

## Reproducibility notes

- Dataset sampling seed and source splits are set by the builder's CLI arguments.
- Input lengths are measured after applying each tokenizer's chat template; tool definitions are passed to the template for `tool_calling` samples.
- The expected target is identified by `metadata.target_message_index` for `agentic_coding` and `validation.target_message_index` for `tool_calling`.
- These are reference-turn length statistics, not model-generated output lengths or speculative-decoding acceptance measurements.
