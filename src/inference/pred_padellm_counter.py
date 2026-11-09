#TRAINING PADELLM COUNTER   
"""

- argparse allows the user to choose a model from the command line using
  --model_name. If no model is given, TinyLlama is used by default.

- transformers loads:
  * AutoTokenizer: converts text into token IDs the model can process.
  * AutoModelForCausalLM: loads a causal language model, which predicts
    the next token using only the tokens before it.

- peft adds LoRA (Low-Rank Adaptation). Instead of training every weight
  in the large base model, LoRA freezes the original weights and trains
  only small adapter matrices.

- The tokenizer may need a padding token when processing batches of
  different-length sequences. If it does not have one, the end-of-sequence
  token is reused as padding.

- The LoRA configuration uses:
  * r=8: the rank of the low-rank update; higher values give more capacity
    but use more memory.
  * lora_alpha=16: scales the strength of the LoRA update.
  * lora_dropout=0.05: randomly disables 5% of LoRA connections during
    training to reduce overfitting.
  * target_modules=["q_proj", "v_proj"]: adds LoRA adapters only to the
    query and value projection layers in attention.
  * bias="none": bias parameters are not trained.
  * task_type="CAUSAL_LM": the model is used for next-token prediction
    and text generation.

- get_peft_model wraps the base model with LoRA so that only the adapter
  parameters are trainable.

"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer,TrainingArguments,DataCollatorForSeq2Seq
from peft import PeftModel
from torch.utils.data import Dataset
from src.data.padellm_entities_count_builder import padellm_counter
from datasets import load_dataset
from pathlib import Path
import json

def load_model_and_adapter(model_name,adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #if tokenize have no pad tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #CausalLM means LM that can only look backwards to predict enxt token
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float32)

    
    model =PeftModel.from_pretrained(model,adapter_dir)
    model.eval()
    return model,tokenizer



def loadDataset():
    dataset = load_dataset("lhoestq/conll2003",split="validation")
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

    return output


def generate_predictions(model, tokenizer, raw_examples,max_new_tokens=256):

    model.eval()
    predictions=[]
    for sentence_id, raw_example in enumerate(raw_examples):


        built_examples = padellm_counter(
            tokens=raw_example["tokens"],
            bio_tags=raw_example["bio_tags"],
            sentence_id=sentence_id,
        )

        for built_example in built_examples:
            result={}
            # Rebuild the exact same prompt format used during training.
            prompt_text = f"{built_example['input']}\n{built_example['task']}"

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Keep only the newly generated tokens, not the prompt.
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_length:]

            generated_pred = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            result["sentence_id"]=sentence_id
            result["tokens"]=raw_example["tokens"]
            result["bio_tags"]=raw_example["bio_tags"]
            result["generation"]=generated_pred
            predictions.append(result)

    return predictions

def save_predictions(predictions, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to: {output_path}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/inference_padellm_counter",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="results/padellm_counter/predictions.jsonl"
    )

    parser.add_argument(
            "--max_new_tokens",
            type=int,
            default=16,
        )


    args=parser.parse_args()
    model, tokenizer = load_model_and_adapter(args.model_name,args.adapter_dir)


    print("Model and tokenizer loaded successfully.")


    
    raw_examples =loadDataset() 

        
    print(f"Running inference on {len(raw_examples)} validation sentences.")
        
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        raw_examples=raw_examples,
        max_new_tokens=args.max_new_tokens,
    )

    save_predictions(predictions=predictions,output_file=args.output_file)
        
      
   
if __name__=="__main__":
    main()