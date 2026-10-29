#TRAINING AUTOREG STRUCT
"""
This module loads a pretrained causal language model and prepares it for
LoRA fine-tuning.

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

This module only loads and prepares the model and tokenizer. It does not
yet train the model, compute loss, save checkpoints, or generate NER
predictions.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer,TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from src.data.autoreg_struct_data_builder import build_autoreg_struct_examples
from datasets import load_dataset
import time

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #if tokenize have no pad tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #CausalLM means LM that can only look backwards to predict enxt token
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,   # r means LoRA rank, higher r means more trainable parameters and more capacity
        lora_alpha=16,   #LoRA scaling factor, controls how strongly the LoRA update affects the frozen base weights
        lora_dropout=0.05,# Dropout applied to LoRA layers during training; helps reduce overfitting
        target_modules=["q_proj","v_proj"],# Model layers that receive LoRA adapters: attention query and value projections
        bias="none",#not train bias parameters; only LoRA adapter matrices are trainable
        task_type="CAUSAL_LM", # Task type: causal language modeling (next-token prediction / text generation)
    )

    model = get_peft_model(model,lora_config)
    model.print_trainable_parameters()
    return model,tokenizer

def build_training_examples(tokenizer,input_text,task_text,output_text,max_length=256):
    prompt_text = f"{input_text}\n{task_text}"

    prompt_tokenization = tokenizer(prompt_text,
                           truncation=True, #if input text(input+tasks) exceed max_length then cut off extra tokens rather than causing errors
                           max_length=max_length,
                           add_special_tokens=True #add model specefic tokens, like beginning of seq etc
                           )
    prompt_ids = prompt_tokenization["input_ids"] #Extract only token ID list from the tokenizer’s output. This is the numerical version of the prompt.

    output_tokenization = tokenizer(output_text,
                           truncation=True, #if input text(input+tasks) exceed max_length then cut off extra tokens rather than causing errors
                           max_length=max_length,
                           add_special_tokens=False #dont add model specefic tokens, like beginning of seq etc
                           )
    output_ids = output_tokenization["input_ids"] + [tokenizer.eos_token_id] #gets target token IDs and appends the end of sequence token. EOS teaches the model to stop after producing the NER output.

    input_ids = prompt_ids+output_ids #combine prompt and output in one sequence: [prompt tokens] + [target tokens] + [EOS]

    '''Create an attention mask with a 1 for every token.
    1 means this is a real token and the model should attend to it.
    This is valid here because the example has no padding'''
    attention_mask = [1]*len(input_ids)

    ''' Create labels for the loss:
    Every prompt token gets -100, so the loss ignores it, as tokens with id -100 are ignored while calculating loss.
    Every output/target token keeps its real token ID,so the model is trained to predict it.
    output_ids already ends with EOS, so the model also learns when to stop.
    This implements loss only on the output tokens.
    '''

    '''
    input_ids = [prompt tokens] + [target tokens] + [EOS]
    labels    = [-100, -100, ..., -100] + [target token IDs] + [EOS]
    '''
    labels= ([-100] * len(prompt_ids)) + output_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }




class OverfitDataset(Dataset):
    def __init__(self,examples):
        self.examples=examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,index):
        return self.examples[index]


#we fine tune LoRA with 20 examples
def create_overfit_examples(raw_examples,tokenizer,num_examples):
    training_examples=[]    

    for sentence_id,raw_example in enumerate(raw_examples):
        built_examples=build_autoreg_struct_examples(tokens=raw_example["tokens"],bio_tags=raw_example["bio_tags"],sentence_id=sentence_id)

        
        for built_example in built_examples:
            training_example = build_training_examples(tokenizer=tokenizer, input_text = built_example["input"],task_text=built_example["task"],output_text=built_example["output"])
            training_examples.append(training_example)

            if len(training_examples)==num_examples:
                return training_examples

    return training_examples

def loadDataset():
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

    return output

# Check whether the model can reproduce the exact target
# for the same examples used in the CPU overfit test.
def test_overfit_examples(model, tokenizer, raw_examples, num_examples=20):

    model.eval()

    correct = 0
    total = 0

    # Use the same first examples and same question rotation as training.
    for sentence_id, raw_example in enumerate(raw_examples):
        if total == num_examples:
            break


        built_examples = build_padellm_examples(
            tokens=raw_example["tokens"],
            bio_tags=raw_example["bio_tags"],
            sentence_id=sentence_id,
        )

        for built_example in built_examples:
            if total == num_examples:
                break

            # Rebuild the exact same prompt format used during training.
            prompt_text = f"{built_example['input']}\n{built_example['task']}"

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )

            # Keep only the newly generated tokens, not the prompt.
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_length:]

            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            expected_text = built_example["output"].strip()
            if expected_text=="<eos>":
                is_correct = generated_text == ""
            else:
                is_correct = generated_text == expected_text
                
            if is_correct:
                correct += 1
            total += 1

            print(f"\nExample {total}")
            print(f"Question: {built_example['task']}")
            print(f"Expected: {expected_text}")
            print(f"Generated: {generated_text}")
            print(f"Correct: {is_correct}")

    print(f"\nOverfit test: {correct}/{total} exact matches")

    return correct, total

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/overfit_test",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
    )

    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)


    args=parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)


    print("Model and tokenizer loaded successfully.")
    #below is a little example show loss masking
    '''
    test_example = build_training_examples(
        tokenizer=tokenizer,
        input_text="Apple hired John Smith in Dublin .",
        task_text="Find the 1st ORG mention.",
        output_text="Apple",
        max_length=256
    )

    
    
    print("Number of input ids:", len(test_example["input_ids"]))
    print("Number of labels:", len(test_example["labels"]))

    print("Input ids:")
    print(test_example["input_ids"])

    print("Labels:")
    print(test_example["labels"])

    print("Decoded full sequence:")
    print(tokenizer.decode(test_example["input_ids"]))

    print("Decoded supervised output:")
    supervised_ids = [
        token_id
        for token_id in test_example["labels"]
        if token_id != -100
    ]
    print(tokenizer.decode(supervised_ids))
    '''

    
    raw_examples =loadDataset() 
    training_examples = create_overfit_examples(
        raw_examples=raw_examples,
        tokenizer=tokenizer,
        num_examples=args.num_examples if args.num_examples is not None else 20,
    )

    if args.num_examples is not None:
        training_examples = training_examples[:args.num_examples]

    dataset = OverfitDataset(training_examples)
    #below is test to check building of training examples before connecting to main dataset
    '''
    print("Number of training examples:", dataset.__len__())

    first_example = dataset.__getitem__(0)

    print("First example input:")
    print(tokenizer.decode(first_example["input_ids"]))

    print("First example supervised output:")
    supervised_ids = [token_id for token_id in first_example["labels"] if token_id != -100] #this will keep all supervised token ids- ids that account for loss amking (ids that are not -100)
    print(tokenizer.decode(supervised_ids))'''

    training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=args.num_epochs,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
    max_steps=args.max_steps,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    training_time_seconds = time.time() - start_time

    print(f"Training time: {training_time_seconds:.1f} seconds")
    print(f"Total training steps: {trainer.state.global_step}")

    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training finished.")
    print(f"Model saved to: {args.output_dir}")

   

if __name__=="__main__":
    main()