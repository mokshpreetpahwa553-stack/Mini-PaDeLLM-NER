from src.data.padellm_data_builder import build_padellm_examples
from src.data.onestep_data_builder import build_onestep_examples
from src.data.autoreg_struct_data_builder import build_autoreg_struct_examples
from datasets import load_dataset

import csv
import numpy as np


# this function calculates input and output word token statistics for one format
def calculate_token_stats(examples):
    input_lengths=[]
    output_lengths=[]

    #for each example we count number of input and output tokens
    for example in examples:
        input_lengths.append(len(example["input"].split()))
        output_lengths.append(len(example["output"].split()))

    return {
        "num_examples": len(examples),
        "input_tokens_mean": np.mean(input_lengths),
        "input_tokens_min": min(input_lengths),
        "input_tokens_max": max(input_lengths),
        "output_tokens_mean": np.mean(output_lengths),
        "output_tokens_min": min(output_lengths),
        "output_tokens_max": max(output_lengths)
    }


dataset = load_dataset("lhoestq/conll2003",split="train")
ner_tag_names = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]

output_of_padellm=[]
output_of_onestep=[]
output_of_autoreg_struct=[]

for i,sentence in enumerate(dataset):
    ner_tags=[]

    #as in dataset ner_tags are in numebr, we made a list at top of file with indexes corresponding to numbers in dataset for ner tags
    #using for loop we convert number to names tag
    for tag_id in sentence["ner_tags"]:
        tag= ner_tag_names[tag_id]
        ner_tags.append(tag)

    tokens = sentence["tokens"]

    output_of_padellm.extend(build_padellm_examples(tokens=tokens,bio_tags=ner_tags,sentence_id=i))
    output_of_autoreg_struct.extend(build_autoreg_struct_examples(tokens=tokens,bio_tags=ner_tags,sentence_id=i))
    output_of_onestep.extend(build_onestep_examples(tokens=tokens,bio_tags=ner_tags,sentence_id=i))



#calculation stats for each format
padellm_stats = calculate_token_stats(output_of_padellm)
autoreg_struct_stats = calculate_token_stats(output_of_autoreg_struct)
onestep_stats = calculate_token_stats(output_of_onestep)

#putting all results in a list so printing is easy
results = [["padellm",padellm_stats], ["autoreg_struct",autoreg_struct_stats],["onestep",onestep_stats]]


# print the dataset sizes and tokenlength statistics
for format_name, stats in results:
    print(f"Format: {format_name}")
    print(f"Number of examples: {stats['num_examples']}")
    print(f"Input tokens - mean: {stats['input_tokens_mean']:.2f}, min: {stats['input_tokens_min']}, max: {stats['input_tokens_max']}")
    print(f"Output tokens - mean: {stats['output_tokens_mean']:.2f}, min: {stats['output_tokens_min']}, max: {stats['output_tokens_max']}")
    print("")

#writing all stats in csv file in results folder
csv_path = "results/dataset_sizes.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    #csv file header
    writer.writerow([
        "format",
        "num_examples",
        "input_tokens_mean",
        "input_tokens_min",
        "input_tokens_max",
        "output_tokens_mean",
        "output_tokens_min",
        "output_tokens_max"
    ])

    # write one row for every format
    for format_name, stats in results:
        writer.writerow([
            format_name,
            stats["num_examples"],
            stats["input_tokens_mean"],
            stats["input_tokens_min"],
            stats["input_tokens_max"],
            stats["output_tokens_mean"],
            stats["output_tokens_min"],
            stats["output_tokens_max"]
        ])