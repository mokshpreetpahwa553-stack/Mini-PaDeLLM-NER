import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from src.data.bio_to_mentions_converter import bio_to_mention_conv

from src.data.padellm_entities_count_builder import padellm_counter


LABELS=["ORG","PER","MISC","LOC"]


def load_dataset_sentences(limit):
    dataset=load_dataset("lhoestq/conll2003", split="train")
    ner_tag_names=["O", "B-PER", "I-PER", "B-ORG", "I-ORG","B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    sentences=[]

    for sentence in dataset:
        bio_tags=[ner_tag_names[tag_id] for tag_id in sentence["ner_tags"]]

        sentences.append({
            "tokens": sentence["tokens"],
            "bio_tags": bio_tags,
        })

        if len(sentences)==limit:
            break

    return sentences


def load_model_and_tokenizer(model_name, adapter_dir):
    tokenizer=AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    tokenizer.padding_side="left"

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )

    model=PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return model,tokenizer


def build_prompt(sentence, label):
    return f"{sentence}\nCount the number of {label} mentions."


def parse_prediction(text):
    text=text.strip()

    if re.fullmatch(r"\d+", text):
        return int(text), "valid"

    return None, "malformed"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num_sentences",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5
    )

    args=parser.parse_args()

    model,tokenizer=load_model_and_tokenizer(args.model_name,args.adapter_dir)

    sentences=load_dataset_sentences(args.num_sentences)

    all_examples=[]

    for sentence_id, sentence_data in enumerate(sentences):
        sentence_text=" ".join(sentence_data["tokens"])

        count_examples=padellm_counter(tokens=sentence_data["tokens"],bio_tags=sentence_data["bio_tags"],sentence_id=sentence_id)

        for example in count_examples:
            all_examples.append({
                "sentence_id": sentence_id,
                "sentence": sentence_text,
                "label": example["label"],
                "count": int(example["output"]),
                "prompt": build_prompt(sentence_text,example["label"])
            })

    print(f"Loaded {len(all_examples)} count-prediction examples.")

    results=[]

    for start in range(0, len(all_examples), args.batch_size):
        batch=all_examples[start : start+args.batch_size]

        prompts=[example["prompt"] for example in batch]

        inputs=tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            output_ids=model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids=output_ids[:,prompt_length:]
        generated_texts=tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        for example, generated_text in zip(batch, generated_texts):
            prediction, status=parse_prediction(generated_text)

            correct=(
                prediction is not None and prediction==example["count"])

            results.append({
                **example,
                "prediction": prediction,
                "raw_prediction": generated_text,
                "status": status,
                "correct": correct,
            })

    print("\nPer-example results:")
    print("sentence_id | label | count | prediction | status  | correct")

    for result in results:
        print(
            f"{result['sentence_id']:11} | "
            f"{result['label']:5} | "
            f"{result['count']:4} | "
            f"{str(result['prediction']):10} | "
            f"{result['status']:7} | "
            f"{result['correct']}"
        )
        print("\nSample raw predictions:")


    total=len(results)
    correct=sum(result["correct"] for result in results)
    malformed=sum(result["status"]=="malformed" for result in results)
    empty=sum(result["raw_prediction"].strip()=="" for result in results)

    print("\nFinal metrics:")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}/{total}")
    print(f"Exact-match accuracy: {correct / total:.2%}")
    print(f"Malformed outputs: {malformed}")
    print(f"Empty outputs: {empty}")


if __name__=="__main__":
    main()