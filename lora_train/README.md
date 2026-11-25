# Train and evaluate an Agata LoRA adapter

Run the commands from the repository root with Python 3.12. The training environment is separate from the API and batch containers.

```sh
python3.12 -m venv .venv-lora
source .venv-lora/bin/activate
python -m pip install -r lora_train/requirements.txt
```

## Prepare data

Use UTF-8 JSONL with one object per line. Each row needs an ISO date, a supported sign and topic, a Russian completion of two or three sentences, and its source. Write the body without a heading. See [training examples](examples/train.jsonl) and [validation examples](examples/validation.jsonl) for the schema. These six original examples exercise the pipeline.

For a quality experiment, prepare a larger original or licensed corpus covering all four topics and all signs. Group near-duplicate texts before splitting and keep the groups in separate splits. The training command rejects exact normalized text overlap between train and validation. Keep the evaluation dates and prompts out of training and tuning. If you use a different held-out set, record it and pass its dates explicitly.

## Train

The default base is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). The first command below resolves the current commit. Record and reuse that full SHA to reproduce the run.

```sh
revision="$(python -c 'from huggingface_hub import model_info; print(model_info("Qwen/Qwen2.5-0.5B-Instruct").sha)')"
python -m lora_train.train \
  --revision "$revision" \
  --train lora_train/examples/train.jsonl \
  --validation lora_train/examples/validation.jsonl \
  --output data/lora/smoke-adapter \
  --max-steps 1 --batch-size 2 --gradient-accumulation 1 --cpu
```

For an experiment, point `--train` and `--validation` at your corpus, choose a new output directory, and use `--epochs` with `--max-steps` omitted. Trainer selects the training device unless you pass `--cpu`. Use `--bf16` on compatible hardware. You can pass a local directory with Qwen-compatible safetensor weights and a chat tokenizer as `--model`.

Training wraps the attention query and value projections with PEFT LoRA. It builds the chat prompt using the same formatter as Agata and masks prompt and padding tokens from the loss. The output contains the adapter, tokenizer, and `training.json` with the command settings, prompt version, dataset hashes, package versions, and losses. The base model remains a separate dependency.

## Generate and rate

```sh
python -m lora_train.evaluate generate \
  --adapter data/lora/smoke-adapter --output data/lora/smoke-eval \
  --dates 2026-12-01 --max-new-tokens 8
```

The command above exercises the evaluator with one date and short outputs. For the actual experiment, use the trained adapter and a fresh output directory, and omit `--dates` and `--max-new-tokens` to use the standard three dates, 144 prompts, and the batch token limit. Choose `--device cpu`, `--device mps`, or `--device cuda`.

- `outputs.jsonl` stores the first attempt and the result after validation, retry, and fallback for every prompt and variant.
- `automatic.json` records the run configuration, raw validation failure rate, retry rate, fallback rate, repeated-text rate within each sign and topic across dates, and generation time for sequential offline inference.
- `blind-raw.jsonl` and `blind-workflow.jsonl` contain shuffled texts to rate. Give reviewers the blind file and matching `ratings-*.csv`. Keep the output and key files with the experiment owner.

For each row, enter `on_topic=1` if the advice addresses the requested theme, or `0` if it is off-topic or vague. For the general theme, broad daily advice qualifies. Rate `naturalness` as `1` for unusable or hard to read, `2` for understandable but awkward, and `3` for idiomatic, readable Russian. Use `1` for empty responses. Rate the raw and workflow stages separately.

```sh
python -m lora_train.evaluate score \
  --run data/lora/smoke-eval --stage raw \
  --ratings data/lora/smoke-eval/ratings-raw.csv
```

After labeling the workflow stage, repeat with `--stage workflow` and its ratings file. Scoring writes topic adherence and average naturalness for all three variants. It reports the paired LoRA-minus-base topic difference in percentage points with a 95% bootstrap interval grouped by sign and topic across dates.

Before the final run, decide what improvement in topic adherence and what changes in naturalness and fallback rate would justify a release. Compare the raw and workflow results and review representative failures. Publish the results from a held-out run alongside the dataset provenance and model revisions.
