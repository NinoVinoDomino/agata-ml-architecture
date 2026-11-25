import argparse
import asyncio
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter

from agata.domain import COMBINATIONS
from agata.generation import (
    PROMPT_VERSION,
    ForecastWorkflow,
    GeneratedText,
    TemplateGenerator,
    editorial_body,
    validate_text,
)
from lora_train.common import chat_messages, normalized, read_jsonl, write_json, write_jsonl

DEFAULT_DATES = (date(2026, 12, 1), date(2026, 12, 2), date(2026, 12, 3))


class LocalGenerator:
    def __init__(self, model, tokenizer, args, set_seed):
        self.id = "offline-model"
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.set_seed = set_seed

    async def generate(self, request):
        key = f"{self.args.seed}/{request.date}/{request.sign}/{request.topic}/{request.attempt}"
        self.set_seed(int.from_bytes(sha256(key.encode()).digest()[:4], "big"))
        inputs = self.tokenizer.apply_chat_template(
            chat_messages(request),
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.args.device)
        output = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()
        return GeneratedText(text, "model")


class RecordFirstAttempt:
    def __init__(self, generator):
        self.generator = generator
        self.id = generator.id
        self.text = ""

    async def generate(self, request):
        candidate = await self.generator.generate(request)
        if request.attempt == 1:
            self.text = candidate.text
        return candidate


async def collect(variant, generator, dates):
    recorded = RecordFirstAttempt(generator)
    workflow = ForecastWorkflow(recorded)
    rows = []
    for day in dates:
        for sign, topic in COMBINATIONS:
            started = perf_counter()
            forecast = await workflow.run(day, sign, topic)
            rows.append(
                {
                    "variant": variant,
                    "date": str(day),
                    "sign": sign,
                    "topic": topic,
                    "raw": recorded.text,
                    "workflow": forecast.text,
                    "raw_issue": validate_text(recorded.text, sign, topic),
                    "source": forecast.source,
                    "attempts": forecast.attempts,
                    "seconds": perf_counter() - started,
                }
            )
    return rows


def automatic_metrics(rows):
    texts_by_pair = defaultdict(list)
    validation_issues = Counter()
    for row in rows:
        texts_by_pair[row["sign"], row["topic"]].append(normalized(row["workflow"]))
        if row["raw_issue"]:
            validation_issues[row["raw_issue"]] += 1
    size = len(rows)
    unique_texts = sum(len(set(texts)) for texts in texts_by_pair.values())
    return {
        "requests": size,
        "raw_validation_issue_rate": validation_issues.total() / size,
        "raw_validation_issues": dict(validation_issues),
        "retry_rate": sum(row["attempts"] > 1 for row in rows) / size,
        "fallback_rate": sum(row["source"] == "fallback" for row in rows) / size,
        "workflow_repeated_text_rate": 1 - unique_texts / size,
        "total_generation_seconds": sum(row["seconds"] for row in rows),
    }


def generate(args):
    if args.output.exists():
        raise ValueError("--output must be a new directory")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    manifest_path = args.adapter / "training.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["prompt_version"] != PROMPT_VERSION:
        raise ValueError("adapter was trained with a different Agata prompt version")
    model_name = manifest["arguments"]["model"]
    revision = manifest["arguments"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, use_safetensors=True)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    generator = LocalGenerator(model, tokenizer, args, set_seed)
    dates = args.dates or DEFAULT_DATES
    if len(set(dates)) != len(dates):
        raise ValueError("evaluation dates must be unique")
    rows = asyncio.run(collect("template", TemplateGenerator(), dates))
    with torch.inference_mode():
        with model.disable_adapter():
            rows.extend(asyncio.run(collect("base", generator, dates)))
        rows.extend(asyncio.run(collect("lora", generator, dates)))

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "outputs.jsonl", rows)
    for stage in ("raw", "workflow"):
        shuffled = rows.copy()
        random.Random(f"{args.seed}/{stage}").shuffle(shuffled)
        blind = []
        key = []
        for number, row in enumerate(shuffled, 1):
            item_id = f"{stage}-{number:04}"
            prompt = {field: row[field] for field in ("date", "sign", "topic")}
            text = editorial_body(row[stage], row["sign"], row["topic"])
            blind.append({"id": item_id, **prompt, "text": text})
            key.append({"id": item_id, **prompt, "variant": row["variant"]})
        write_jsonl(args.output / f"blind-{stage}.jsonl", blind)
        write_jsonl(args.output / f"key-{stage}.jsonl", key)
        with (args.output / f"ratings-{stage}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "on_topic", "naturalness"])
            writer.writerows((row["id"], "", "") for row in blind)

    write_json(
        args.output / "automatic.json",
        {
            "model": model_name,
            "revision": revision,
            "prompt_version": PROMPT_VERSION,
            "training_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "adapter_weights_sha256": sha256(
                (args.adapter / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "device": args.device,
            "seed": args.seed,
            "dates": [str(day) for day in dates],
            "decoding": {
                "temperature": 0.7,
                "top_k": 0,
                "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens,
            },
            "variants": {
                label: automatic_metrics([row for row in rows if row["variant"] == label])
                for label in ("template", "base", "lora")
            },
        },
    )


def score(args):
    key = {row["id"]: row for row in read_jsonl(args.run / f"key-{args.stage}.jsonl")}
    values = {}
    with args.ratings.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            item_id = row["id"]
            if item_id not in key or item_id in values:
                raise ValueError(f"unknown or duplicate rating ID: {item_id}")
            topic = int(row["on_topic"])
            naturalness = int(row["naturalness"])
            if topic not in (0, 1) or not 1 <= naturalness <= 3:
                raise ValueError(f"{item_id}: on_topic must be 0/1 and naturalness 1–3")
            values[item_id] = (topic, naturalness)
    if set(values) != set(key):
        raise ValueError("provide a rating for every ID in this stage")

    paired = defaultdict(dict)
    summaries = defaultdict(list)
    for item_id, row in key.items():
        variant = row["variant"]
        summaries[variant].append(values[item_id])
        paired[row["sign"], row["topic"], row["date"]][variant] = values[item_id]

    clusters = defaultdict(list)
    for (sign, topic, _), variants in paired.items():
        clusters[sign, topic].append(variants["lora"][0] - variants["base"][0])
    cluster_means = [mean(deltas) for deltas in clusters.values()]
    rng = random.Random(42)
    estimates = sorted(
        100 * mean(rng.choices(cluster_means, k=len(cluster_means))) for _ in range(5000)
    )
    summary = {
        "stage": args.stage,
        "paired_prompts": len(paired),
        "variants": {
            name: {
                "on_topic_rate": mean(topic for topic, _ in ratings),
                "mean_naturalness": mean(naturalness for _, naturalness in ratings),
            }
            for name, ratings in summaries.items()
        },
        "lora_minus_base_topic_percentage_points": 100 * mean(cluster_means),
        "bootstrap_95_interval_percentage_points": [estimates[125], estimates[4874]],
    }
    write_json(args.run / f"human-{args.stage}.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare Agata templates, base model and LoRA.")
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("generate")
    inference.add_argument("--adapter", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--dates", nargs="+", type=date.fromisoformat)
    inference.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    inference.add_argument("--seed", type=int, default=42)
    inference.add_argument("--max-new-tokens", type=int, default=240)
    scoring = commands.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--stage", choices=("raw", "workflow"), required=True)
    scoring.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate(args)
        else:
            score(args)
    except (ImportError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
