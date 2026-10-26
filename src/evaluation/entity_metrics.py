from src.data.bio_to_mentions_converter import bio_to_mention_conv
"""Here, we check our evaluation system before developing a model"""
"""
FOR TESTING
gold_mentions - output from converter
handmade list-  manual amde output of type of output from converter
this function takes input from bio_to_mention converter and 
handmade list of label -> mention text, and we manipulate our handmade input
to see if our evaluaiton system works correctly


FOR REAL
gold mentions are output from converter
predicted_mentions are predictions from model"""

"""
It outputs
precision, recall , f1, true positive, false positive, false negative"""

def compute_micro_entity_metrics(gold_mentions , predicted_mentions):
    true_positive,false_positive,false_negative=0,0,0
    unused_gold=list(gold_mentions)

    for pred in predicted_mentions:
        matched=False
        for i,gold in enumerate(unused_gold):
            
            if gold ==pred:
                true_positive+=1
                unused_gold.pop(i)
                matched=True
                break

        if matched==False:
            false_positive+=1

    false_negative = len(unused_gold)

    precision = true_positive / (true_positive+false_positive) if (true_positive+false_positive)>0 else 0.0
    recall = true_positive / (true_positive+false_negative) if (true_positive+false_negative) else 0.0
    f1 = (2*precision*recall) / (precision+recall) if (precision+recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives" : true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative
    }


#above function works with single snetence, but we need funciton taht would work for whole datase
def evaluate_dataset(all_examples , all_predicted_mentions):
    total_true_positive,total_false_positive,total_false_negative=0,0,0

    for example,predicted_mention in zip(all_examples , all_predicted_mentions):
        tokens = example["tokens"]
        bio_tags=example["bio_tags"]

        gold_mentions = bio_to_mention_conv(tokens=tokens,tags=bio_tags)
        metrics = compute_micro_entity_metrics(gold_mentions=gold_mentions,predicted_mentions=predicted_mention)

        total_true_positive+=metrics["true_positives"]
        total_false_positive+=metrics["false_positives"]
        total_false_negative+=metrics["false_negatives"]


    precision = total_true_positive / (total_true_positive+total_false_positive) if (total_true_positive+total_false_positive)>0 else 0.0
    recall = total_true_positive / (total_true_positive+total_false_negative) if (total_true_positive+total_false_negative) else 0.0
    f1 = (2*precision*recall) / (precision+recall) if (precision+recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives" : total_true_positive,
        "false_positives": total_false_positive,
        "false_negatives": total_false_negative
    }




        

def helper(tokens,bio_tags,predicted_mentions):
    gold_mentions = bio_to_mention_conv(tokens=tokens,tags=bio_tags)

    metrics = compute_micro_entity_metrics(gold_mentions,predicted_mentions)

    return metrics


print(helper(tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."], bio_tags= ["B-ORG","O","B-PER","I-PER","O","B-LOC", "O"], predicted_mentions =  [("ORG", "Apple"), ("PER", "John Smith"), ("LOC", "Dublin"), ("MISC", "German")]))
print(helper(tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."], bio_tags= ["B-ORG","O","B-PER","I-PER","O","B-LOC", "O"], predicted_mentions =  [("ORG", "Apple"), ("PER", "John Smith"), ("LOC", "Dublin")]))
print(helper(tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."], bio_tags= ["B-ORG","O","B-PER","I-PER","O","B-LOC", "O"], predicted_mentions =  [("ORG", "Apple"), ("PER", "John Smith")]))
print(helper(tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."], bio_tags= ["B-ORG","O","B-PER","I-PER","O","B-LOC", "O"], predicted_mentions =  [("ORG", "Apple"), ("ORG", "John Smith"), ("LOC", "Dublin")]))
print(helper(tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."], bio_tags= ["B-ORG","O","B-PER","I-PER","O","B-LOC", "O"], predicted_mentions =  []))
print(helper(tokens=["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."], bio_tags=["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"], predicted_mentions=[("PER","Elon Musk"), ("PER", "SpaceX")]) )
print(helper(tokens=["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."], bio_tags=["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"], predicted_mentions=[("PER","Elon Musk"), ("ORG", "SpaceX")]) )
print(helper(tokens=["The", "conference", "will", "be", "held", "in", "Paris", "on", "Monday", "."],bio_tags=["O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-DATE", "O"], predicted_mentions=[("ORG","Conference"),("LOC","Paris"),("ORG","Paris"),("DATE","MONDAY")]))
print(helper(tokens=["The", "conference", "will", "be", "held", "in", "Paris", "on", "Monday", "."],bio_tags=["O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-DATE", "O"], predicted_mentions=[("ORG","Conference"),("LOC","Paris"),("ORG","Paris"),("DATE","Monday")]))
print(helper(tokens=["The", "conference", "will", "be", "held", "in", "Paris", "on", "Monday", "."],bio_tags=["O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-DATE", "O"], predicted_mentions=[("LOC","Paris"),("DATE","Monday")]))

