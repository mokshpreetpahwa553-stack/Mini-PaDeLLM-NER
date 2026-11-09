from src.data.padellm_deduplication import de_deplication


def test_keeps_single_prediction():
    predictions = [{
            "sentence_id":0,
            "label":"ORG",
            "mention":"EU",
            "prediction":"EU",
            "probability":0.90,
            "mention_index":1,
            "status":"valid"
        }]
    result =de_deplication(predictions)

    assert len(result)==1
    assert result[0]["label"]=="ORG"
    assert result[0]["mention"]=="EU"
    assert result[0]["probability"]==0.90


def test_keeps_highest_probability_for_duplicate_mention():
    predictions = [{
            "sentence_id":0,
            "label":"ORG",
            "mention":"German",
            "prediction":"German",
            "probability":0.70,
            "mention_index":1,
            "status":"valid"},
        {
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.95,
            "mention_index":1,
            "status":"valid"
        }]

    result= de_deplication(predictions)

    assert len(result)==1
    assert result[0]["label"]=="MISC"
    assert result[0]["probability"]==0.95


def test_keeps_different_mentions_in_same_sentence():
    predictions = [{
            "sentence_id":0,
            "label":"ORG",
            "mention":"EU",
            "prediction":"EU",
            "probability":0.90,
            "mention_index":1,
            "status":"valid"},
        {
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.80,
            "mention_index":1,
            "status":"valid"
        }]

    result =de_deplication(predictions)

    assert len(result)==2

    mentions= {prediction["mention"] for prediction in result}
    assert mentions=={"EU", "German"}


def test_keeps_same_mention_in_different_sentences():
    predictions=[{
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.95,
            "mention_index":1,
            "status":"valid"
        },
        {
            "sentence_id":1,
            "label":"ORG",
            "mention":"German",
            "prediction":"German",
            "probability":0.70,
            "mention_index":1,
            "status":"valid" }
    ]

    result =de_deplication(predictions)

    assert len(result)==2

def test_ignores_none_mention():
    predictions = [
        {
            "sentence_id":0,
            "label":"ORG",
            "mention":None,
            "prediction":"",
            "probability":0.90,
            "mention_index":1,
            "status":"empty"
        },
        {
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.80,
            "mention_index":1,
            "status":"valid"
        }
    ]

    result = de_deplication(predictions)

    assert len(result)==1
    assert result[0]["mention"]=="German"


def test_ignores_empty_mention():
    predictions = [{
            "sentence_id":0,
            "label":"ORG",
            "mention":"",
            "prediction":"",
            "probability":0.90,
            "mention_index":1,
            "status":"empty"
        },{
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.80,
            "mention_index":1,
            "status":"valid"
        }
    ]

    result =de_deplication(predictions)

    assert len(result)==1
    assert result[0]["mention"]=="German"


def test_handles_equal_probabilities():
    predictions = [
        {
            "sentence_id":0,
            "label":"ORG",
            "mention":"German",
            "prediction":"German",
            "probability":0.80,
            "mention_index":1,
            "status":"valid"
        },
        {
            "sentence_id":0,
            "label":"MISC",
            "mention":"German",
            "prediction":"German",
            "probability":0.80,
            "mention_index":1,
            "status":"valid"
        }
    ]

    result = de_deplication(predictions)

    assert len(result)==1
    assert result[0]["label"]=="ORG"

