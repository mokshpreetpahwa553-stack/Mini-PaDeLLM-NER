from src.data.bio_to_mentions_converter import bio_to_mention_conv
"""this module, will convert whole database in a way that works as input for OneStep"""


"""
For each label in each sentence, we create one example:
Apple hired John Smith in Dublin.


Gold mentions:
("ORG", "Apple")
("PER", "John Smith")
("LOC", "Dublin")


Create four OneStep examples:


Input:  Apple hired John Smith in Dublin .
Task:   Find all ORG mentions.
Output: Apple


Input:  Apple hired John Smith in Dublin .
Task:   Find all PER mentions.
Output: John Smith


Input:  Apple hired John Smith in Dublin .
Task:   Find all MISC mentions.
Output: <none>


Input:  Apple hired John Smith in Dublin .
Task:   Find all LOC mentions.
Output: Dublin


If a sentence had two PER mentions:
Find all PER mentions. -> John Smith <sep> Mary Byrne


number of OneStep examples = number of labels in each sentence
"""
def build_onestep_examples(tokens, bio_tags, sentence_id):
    output = []
    gold_mentions = bio_to_mention_conv(tokens,bio_tags)
    sentence = " ".join(tokens)

    ner_labels_and_mentions={"ORG": [],"PER": [], "MISC": [], "LOC": []}

    for gold_mention in gold_mentions:
            label=gold_mention[0]
            entity_text = gold_mention[1]
            ner_labels_and_mentions[label].append(entity_text)

    for label,mentions in ner_labels_and_mentions.items():
        dic={}
        dic["input"]=sentence
        dic["task"]=f"Find all {label} mentions."

        if len(mentions)==0:
            dic["output"]="<none>"
        else:
            dic["output"] = " <sep> ".join(mentions)

        dic["format"]="onestep"
        dic["sentence_id"]=sentence_id

        output.append(dic)

    return output
    