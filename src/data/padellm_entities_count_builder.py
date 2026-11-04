from src.data.bio_to_mentions_converter import bio_to_mention_conv
'''
this module  will create examples for coount, like count number of PER entites
'''
def padellm_counter(tokens,bio_tags,sentence_id):
    gold_mentions  = bio_to_mention_conv(tokens,bio_tags)

    labels_count = {"ORG":0,"PER":0,"MISC":0,"LOC":0}
    sentence = " ".join(tokens)

    for label,entity in gold_mentions:
        labels_count[label]+=1

    output=[]
    for label,count in labels_count.items():
        output.append({
            "input":sentence,
            "label": label,
            "task":f"Count the number of {label} mentions.",
            "output": str(count),
            "sentence_id":sentence_id
        })

    return output

    
