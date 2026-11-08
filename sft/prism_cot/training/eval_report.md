| model | figstep | mmsafety | jailbreak avg | siuo | mossbench refusal↓ |
|---|---|---|---|---|---|
| qwen2vl-7b-base | 44.60 | 57.77 | 51.19 | 16.77 | 2.67 |
| prism-sft | 98.60 | 94.46 | 96.53 | 30.54 | 30.33 |
| prism-sft-lora-repro | 97.60 | 94.88 | 96.24 | 31.74 | 22.00 |

| model | bench | n | judge_fail | score | kw_refusal | truncated | no_output_tag |
|---|---|---|---|---|---|---|---|
| qwen2vl-7b-base | figstep | 500 | 0 | 44.6 | 21.6 | 0 | 0 |
| qwen2vl-7b-base | mmsafety | 1680 | 1 | 57.77 | 16.25 | 9 | 0 |
| qwen2vl-7b-base | siuo | 167 | 0 | 16.77 | 8.38 | 2 | 0 |
| qwen2vl-7b-base | mossbench | 300 | 0 | 2.67 | 1.0 | 4 | 0 |
| prism-sft | figstep | 500 | 0 | 98.6 | 81.8 | 0 | 0 |
| prism-sft | mmsafety | 1680 | 0 | 94.46 | 57.74 | 0 | 0 |
| prism-sft | siuo | 167 | 0 | 30.54 | 16.77 | 3 | 2 |
| prism-sft | mossbench | 300 | 0 | 30.33 | 29.67 | 2 | 1 |
| prism-sft-lora-repro | figstep | 500 | 0 | 97.6 | 74.2 | 0 | 0 |
| prism-sft-lora-repro | mmsafety | 1680 | 0 | 94.88 | 56.96 | 1 | 0 |
| prism-sft-lora-repro | siuo | 167 | 0 | 31.74 | 17.37 | 0 | 1 |
| prism-sft-lora-repro | mossbench | 300 | 0 | 22.0 | 20.67 | 5 | 3 |
