from src.data.bio_to_mentions_converter import bio_to_mention_conv
"""this module, will convert whole database in a way that works as input for AutoReg Struct"""

"""
For each sentence, we create one example:
Apple hired John Smith in Dublin.


Gold mentions:
("ORG", "Apple")
("PER", "John Smith")
("LOC", "Dublin")

Create one AutoReg-Struct example:

Input:  Apple hired John Smith in Dublin .
Task:   Find all entity mentions in the sentence.
Output: ORG: Apple <sep> PER: John Smith <sep> LOC: Dublin

If a sentence has no mentions for a label:
Output: MISC: <none>

If a sentence had two PER mentions:
PER: John Smith <sep>  Mary Byrne

number of AutoReg-Struct examples = number of sentences

output should be in form: ORG: ... <sep> PER: ... <sep> MISC: ... <sep> LOC: ...
so for exmaple wiht 2 PER, 3 MISC, 0 ORG, and 1 LOC output would look like
ORG: <none> <sep> PER: John Smith <sep> Mary Byrne <sep> MISC: Apple <sep> Google <sep> Microsoft <sep> LOC: Dublin
"""

def build_autoreg_struct_examples(tokens, bio_tags, sentence_id):
    output = []
    gold_mentions = bio_to_mention_conv(tokens, bio_tags)
    sentence = " ".join(tokens)
    dic = {}
    dic["input"] = sentence
    dic["task"]="Find all entity mentions in the sentence."
    dic["format"]="autoreg-struct"
    dic["sentence_id"]=sentence_id
    ner_labels_and_mentions={"ORG": [],"PER": [], "MISC": [], "LOC": []}
    mentions = []
    for gold_mention in gold_mentions:
        label=gold_mention[0]
        entity_text = gold_mention[1]
        ner_labels_and_mentions[label].append(entity_text)

    final_out=[]
    for label,mention in ner_labels_and_mentions.items():
        if len(mention)==0:
            mentions=f'{label}: <none>'
        else:
            mentions = f'{label}: {" <sep> ".join(ner_labels_and_mentions[label])}'

        final_out.append(mentions)

    dic["output"] = " <sep> ".join(final_out)
    output.append(dic)

    return output


    

    
