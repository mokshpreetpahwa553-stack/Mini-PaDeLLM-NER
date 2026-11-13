#TRAINING PADELLM DATA BUILDER
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
from src.data.padellm_data_builder import build_padellm_examples

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer,TrainingArguments,DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from datasets import load_dataset

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #if tokenize have no pad tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #CausalLM means LM that can only look backwards to predict enxt token
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float32,device_map="cuda")
    model.config.pad_token_id = tokenizer.pad_token_id
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




class TokenizedCausalLMDataset(Dataset):
    def __init__(self,examples):
        self.examples=examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,index):
        return self.examples[index]


def create_data_for_training(raw_examples,tokenizer):
    counter_examples=[]    
  

    for sentence_id,raw_example in enumerate(raw_examples):
        built_examples=build_padellm_examples(tokens=raw_example["tokens"],bio_tags=raw_example["bio_tags"],sentence_id=sentence_id)
        
        
        for built_example in built_examples:
            training_example = build_training_examples(tokenizer=tokenizer, input_text = built_example["input"],task_text=built_example["task"],output_text=built_example["output"])
            counter_examples.append(training_example)

       


    return counter_examples

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
        default="results/padellm_data_builder",
    )
    
    parser.add_argument(
                "--num_epochs",
                type=int,
                default=2,  )
  
    parser.add_argument("--max_steps", type=int, default=-1)


    args=parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)


    print("Model and tokenizer loaded successfully.")


    
    raw_examples =loadDataset() 
    training_examples = create_data_for_training(raw_examples,tokenizer)
    dataset = TokenizedCausalLMDataset(training_examples)

    #for padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model,padding=True,label_pad_token_id=-100,return_tensors="pt")
   
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    logging_steps=1,
    save_steps=50,
    save_strategy="steps",
    report_to=[],
    remove_unused_columns=False,
    dataloader_num_workers=0,
    max_steps=args.max_steps,
    num_train_epochs=args.num_epochs
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
   
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__=="__main__":
    main()