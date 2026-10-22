from src.data.bio_to_mentions_converter import bio_to_mention_conv
def test_bio_to_mentions_basic():
    tokens=["Apple", "hired","John", "Smith", "in", "Dublin","."]
    tags=["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]
    result = bio_to_mention_conv(tokens,tags)
    assert result == [("ORG", "Apple"),("PER", "John Smith"),("LOC", "Dublin"),]

def test_mention_at_end_saved():
    tokens=["Barack", "Obama"]
    tags = ["B-PER", "I-PER"]
    result=bio_to_mention_conv(tokens,tags)
    assert result == [("PER", "Barack Obama")]

def test_for_seperate_same_type_entity():
    tokens=["john","smith"]
    tags=["B-PER", "B-PER"]
    result=bio_to_mention_conv(tokens,tags)
    assert result == [("PER", "john"), ("PER", "smith")]

def test_for_empty_lists():
    assert bio_to_mention_conv([],[]) == []

def test_overall_test():
    tokens= ["Apple", "hired", "John", "Smith", "in", "Dublin", ".","Barack", "Obama", "visited", "UCD", "and", "Trinity", "College","during", "the", "Dublin", "Marathon", "."]
    tags= ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-PER", "I-PER", "O", "B-ORG", "O", "B-ORG", "I-ORG","O", "O", "B-MISC", "I-MISC", "O"]

    result = bio_to_mention_conv(tokens,tags)

    assert result == [("ORG", "Apple"),("PER", "John Smith"),("LOC", "Dublin"),("PER", "Barack Obama"),("ORG", "UCD"),("ORG", "Trinity College"),("MISC", "Dublin Marathon")]

def test_no_entities():
    tokens =["The", "cat", "sat"]
    tags= ["O", "O", "O"]
    result=bio_to_mention_conv(tokens, tags)
    assert result == []

def test_invalid_tagI_start_new_mention():
    tokens = ["John", "Smith", "went", "home", "again"]
    tags =["B-PER", "I-PER", "O", "O", "I-PER"]

    result= bio_to_mention_conv(tokens, tags)

    assert result == [("PER", "John Smith"),("PER", "again")]

