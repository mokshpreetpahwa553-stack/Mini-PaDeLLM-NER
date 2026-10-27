from src.data.bio_to_mentions_converter import bio_to_mention_conv
"""this module, will conver whole database in a way that works a sinput for PaDeLLM"""

"""
For each gold mention, we create one example:
Apple hired John Smith in Dublin.

Gold mentions:
("ORG", "Apple")
("PER", "John Smith")
("LOC", "Dublin")

Create three PaDeLLM examples:

Input:  Apple hired John Smith in Dublin .
Task:   Find the 1st ORG mention.
Output: Apple

Input:  Apple hired John Smith in Dublin .
Task:   Find the 1st PER mention.
Output: John Smith

Input:  Apple hired John Smith in Dublin .
Task:   Find the 1st LOC mention.
Output: Dublin


If a sentence had two PER mentions:
Find the 1st PER mention. → first person
Find the 2nd PER mention. → second person
Also create one extra example for each label that has zero mentions:

Input:  Apple hired John Smith in Dublin .
Task:   Find the 1st MISC mention.
Output: <eos>

number of PaDeLLM examples =number of gold mentions+ number of labels with zero mentions in that sentence
"""

def get_ordinal(index):
    if index==1:
        return "1st"
    if index==2:
        return "2nd"
    if index==3:
        return "3rd"
    return f"{index}th"


def build_padellm_examples(tokens, bio_tags,sentence_id):
    output = []
    gold_mentions = bio_to_mention_conv(tokens,bio_tags)
    sentence = " ".join(tokens)

    ner_labels_and_counts={"ORG": 0,"PER": 0, "MISC": 0, "LOC": 0}
    for gold_mention in gold_mentions:

        dic={}
        dic["input"] = sentence
        label = gold_mention[0]
        ner_labels_and_counts[label]+=1
        dic["task"] = f"Find the {get_ordinal(ner_labels_and_counts[label])} {label} mention."
        dic["output"] = gold_mention[1]
        dic["format"] = "padellm"
        dic["sentence_id"]=sentence_id

        output.append(dic)

    for label,count in ner_labels_and_counts.items():
        if count==0:
            dic={}
            dic["input"]=sentence
            dic["task"] = f"Find the 1st {label} mention."
            dic["output"] = "<eos>"
            dic["format"]="padellm"
            dic["sentence_id"]=sentence_id
            output.append(dic)

    return output
