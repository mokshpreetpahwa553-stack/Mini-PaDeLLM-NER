"""
This script combines step 1(counting), and step 2(mention prompts) to generate full end to end pipeline.
It load weights for adapter_model.safetensors for lora overfitting of counter and data builder
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from datasets import load_dataset
import re
import json
from src.data.padellm_data_builder import get_ordinal
import math
import statistics
import torch
import torch.nn.functional as F
from src.data.padellm_deduplication import de_deplication


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

def padellm_model(model_name,count_adapter_dir,mention_adapter_dir):
    count_model,count_tokenizer=load_model_and_tokenizer(model_name,count_adapter_dir)
    mention_model,mention_tokenizer=load_model_and_tokenizer(model_name,mention_adapter_dir)

    return {
    "count_model": count_model,
    "count_tokenizer": count_tokenizer,
    "mention_model": mention_model,
    "mention_tokenizer": mention_tokenizer,
    }

def load_dataset_sentences(limit):
    dataset = load_dataset("lhoestq/conll2003",split="train")
    ner_tag_names = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]
    output=[]
    for sentence in dataset:
        ner_tags=[]
        dic={}

        #as in dataset ner_tags are in numebr, we made a list at top of file with indexes corresponding to numbers in dataset for ner tags
        #using for loop we convert number to names tag
        for tag_id in sentence["ner_tags"]:
            tag= ner_tag_names[tag_id]
            ner_tags.append(tag)
        
        tokens = sentence["tokens"]
        dic["tokens"]=tokens
        dic["bio_tags"]=ner_tags
        output.append(dic)
        
        if len(output)==limit:
            break

    return output


def build_prompt_for_count(sentence, label):
    return f"{sentence}\nCount the number of {label} mentions."

def build_prompt_for_mention(sentence, mention_index, label):
    return f"{sentence}\nFind the {get_ordinal(mention_index)} {label} mention"


def generate_batch(model,tokenizer,prompts,batch_size,max_new_tokens):
    outputs = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id
            )

        prompt_len=inputs["input_ids"].shape[1]
        new_tokens_ids = generated.sequences[:, prompt_len:]

        for example_idx in range(len(batch_prompts)):
            generated_text=tokenizer.decode(new_tokens_ids[example_idx],
                                            skip_special_tokens=True).strip()

            probability =compute_probability_of_predictions(generated.scores,new_tokens_ids[example_idx],example_idx)
    

            outputs.append({
                "probability":probability,
                "prediction": generated_text
            })

    return outputs


def run_count_model(c_model,c_tokenizer,count_examples,args):
    prompts=[example["prompt"] for example in count_examples]

    raw_outputs=generate_batch(
        c_model,
        c_tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )

    predicted_counts = {}

    for example, raw_output in zip(count_examples, raw_outputs):
        label = example["label"]
        prediction=raw_output["prediction"]
        count, status = parse_count_prediction(prediction)

        predicted_counts[label]= {
            "count": count,
            "raw_output": prediction,
            "probability":raw_output["probability"],
            "status": status
        }

    return predicted_counts

def run_mention_model(m_model,m_tokenizer,mention_examples,args):
    if len(mention_examples)==0:
        return []

    prompts=[example["prompt"] for example in mention_examples]
    raw_outputs = generate_batch(
        m_model,
        m_tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    mention_predictions = []

    for example, raw_output in zip(mention_examples, raw_outputs):
        probability = raw_output["probability"]
        predicted_text=raw_output["prediction"]
        mention, status = parse_mention_prediction(predicted_text)

        mention_predictions.append({
            "sentence_id": example["sentence_id"],
            "label": example["label"],
            "mention_index": example["mention_index"],
            "mention": mention,
            "prediction": predicted_text,
            "status": status,
            "probability":probability
        })

    return mention_predictions

'''it is sequential probability, so like scores contain prob in sequence of each token in token_ids, example_idx 
index of example for whose output we are computing prob for '''
def compute_probability_of_predictions(scores,token_ids,example_idx):
    log_probs=[]
    for step,token_id in enumerate(token_ids):
        prob=scores[step][example_idx]
        log_prob=F.log_softmax(prob,dim=-1)[token_id].item()
        log_probs.append(log_prob)

    if len(log_probs)==0:
        return 0.0

    avg_log_prob = statistics.mean(log_probs)

    return math.exp(avg_log_prob)


def parse_count_prediction(text):
    text=text.strip()

    if re.fullmatch(r"\d+", text):
        return int(text), "valid"

    return None, "malformed"


def parse_mention_prediction(text):
    text = text.strip()

    if text == "":
        return None, "empty"

    if re.fullmatch(r"\d+", text):
        return None, "malformed"

    return text, "valid"


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
    "--model_name",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    parser.add_argument(
    "--count_adapter_dir",
    type=str,
    required=True
    )

    parser.add_argument(
        "--mention_adapter_dir",
        type=str,
        required=True
    )

    parser.add_argument(
    "--num_sentences",
    type=int,
    default=20
)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=15
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output/padellm_end_to_end_predictions.json"
    )

    args=parser.parse_args()
    models=padellm_model(model_name=args.model_name,count_adapter_dir=args.count_adapter_dir,mention_adapter_dir=args.mention_adapter_dir)

    examples=load_dataset_sentences(args.num_sentences)
    
    LABELS=["ORG","PER","MISC","LOC"]

    all_predictions=[]
    for sentence_id,sentence_data in enumerate(examples):
        count_examples=[]
        tokens=sentence_data["tokens"]
        sentence=" ".join(tokens)
        for label in LABELS:
            count_examples.append({
                "sentence_id":sentence_id,
                "label":label,
                "prompt": build_prompt_for_count(sentence,label)
            })
        predicted_counts=run_count_model(models["count_model"],models["count_tokenizer"],count_examples,args)


        mention_examples=[]
        for label in LABELS:
            count=predicted_counts[label]["count"]
            if count is None:
                continue
            for mention_index in range(1,count+1):
                mention_examples.append({
                    "sentence_id":sentence_id,
                    "label":label,
                    "mention_index":mention_index,
                    "prompt":build_prompt_for_mention(sentence,mention_index,label)
                })

        predicted_mentions=run_mention_model(models["mention_model"],models["mention_tokenizer"],mention_examples,args)
        predicted_mentions=de_deplication(predicted_mentions)
        all_predictions.extend(predicted_mentions)



    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_predictions)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()