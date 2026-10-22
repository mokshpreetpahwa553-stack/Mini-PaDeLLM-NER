from datasets import load_dataset
from src.data.bio_to_mentions_converter import bio_to_mention_conv

ner_tag_names = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]

def load_conll_split(split): #split is name of dataset kind like training or test dataset

    dataset = load_dataset("lhoestq/conll2003", split=split)
    output=[]

    for i in range(len(dataset)):
        each_example_formulation={}
        current_example=dataset[i]
        each_example_formulation["id"] = f"{split}_{i:04d}"
        each_example_formulation["tokens"] = current_example["tokens"]
        each_example_formulation["ner_tags"] = []

        #as in dataset ner_tags are in numebr, we made a list at top of file with indexes corresponding to numbers in dataset for ner tags
        #using for loop we convert number to names tag
        for j in range(len(dataset[i]["ner_tags"])):
            num_to_tag = ner_tag_names[current_example["ner_tags"][j]]
            each_example_formulation["ner_tags"].append(num_to_tag)
        each_example_formulation["mentions"] = bio_to_mention_conv(each_example_formulation["tokens"],each_example_formulation["ner_tags"])

        output.append(each_example_formulation)

    return output

output=load_conll_split("train")
print(output[0])
