import argparse
import platform
import re
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from agata.generation import PROMPT_VERSION
from lora_train.common import DEFAULT_MODEL, chat_messages, normalized, read_examples, write_json


def tokenize_example(tokenizer, example, max_length):
    prompt_ids = tokenizer.apply_chat_template(
        chat_messages(example.request()), tokenize=True, add_generation_prompt=True
    )
    answer_ids = tokenizer.encode(example.text, add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError(
            f"{example.date}/{example.sign}/{example.topic}: exceeds --max-length {max_length}"
        )
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PEFT LoRA adapter for Agata.")
    parser.add_argument("--train", type=Path, required=True, help="training JSONL")
    parser.add_argument("--validation", type=Path, required=True, help="validation JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new adapter directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="40-character Hugging Face commit SHA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if not Path(args.model).is_dir() and not re.fullmatch("[0-9a-f]{40}", args.revision or ""):
        parser.error("a Hub model requires --revision with a full commit SHA")
    if args.output.exists():
        parser.error("--output must be a new directory")

    train_examples = read_examples(args.train)
    validation_examples = read_examples(args.validation)
    if {normalized(row.text) for row in train_examples} & {
        normalized(row.text) for row in validation_examples
    }:
        parser.error("training and validation contain identical normalized texts")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        parser.error(f"install lora_train/requirements.txt ({exc})")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.chat_template is None or tokenizer.eos_token_id is None:
        parser.error("the tokenizer must have a chat template and an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data = [tokenize_example(tokenizer, row, args.max_length) for row in train_examples]
    validation_data = [
        tokenize_example(tokenizer, row, args.max_length) for row in validation_examples
    ]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
        ),
    )
    if args.revision:
        model.peft_config["default"].revision = args.revision
    model.print_trainable_parameters()

    args.output.mkdir(parents=True)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            seed=args.seed,
            data_seed=args.seed,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            label_names=["labels"],
            use_cpu=args.cpu,
            bf16=args.bf16,
        ),
    )
    train_metrics = trainer.train().metrics
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.output))
    write_json(
        args.output / "training.json",
        {
            "arguments": vars(args)
            | {
                "train": str(args.train),
                "validation": str(args.validation),
                "output": str(args.output),
            },
            "prompt_version": PROMPT_VERSION,
            "datasets": {
                name: {"rows": len(rows), "sha256": sha256(path.read_bytes()).hexdigest()}
                for name, path, rows in (
                    ("train", args.train, train_examples),
                    ("validation", args.validation, validation_examples),
                )
            },
            "metrics": {"train": train_metrics, "validation": validation_metrics},
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("torch", "transformers", "peft", "accelerate")
            },
        },
    )


if __name__ == "__main__":
    main()
