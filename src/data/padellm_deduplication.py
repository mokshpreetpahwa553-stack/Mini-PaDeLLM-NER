'''
this script contains code that thet removes duplicate mentions for a same sentence
'''

def de_deplication(predictions_dic):
    grouped_sentences_according_to_sentenceid_and_mention={}

    for prediction in predictions_dic:
        sentence_id=prediction["sentence_id"]
        mention=prediction["mention"]
        if mention=="" or mention is None:
            continue
        lst=grouped_sentences_according_to_sentenceid_and_mention.get((sentence_id,mention) , [])

        lst.append({
            "sentence_id":sentence_id,
            "label": prediction["label"],
            "mention":mention,
            "probability":prediction["probability"],
            "mention_index": prediction["mention_index"],
            "prediction":prediction["prediction"],
            "status": prediction["status"],
            })
        grouped_sentences_according_to_sentenceid_and_mention[(sentence_id, mention)]=lst

    deduplicated_predictions=[]

    for predictions in grouped_sentences_according_to_sentenceid_and_mention.values():
        best_prediction = max(predictions,key=lambda prediction:prediction["probability"])

        deduplicated_predictions.append(best_prediction)

    return deduplicated_predictions

