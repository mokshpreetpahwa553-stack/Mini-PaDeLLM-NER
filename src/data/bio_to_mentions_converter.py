import warnings
#funciton to convert tags to BIO to mentions labels
#B in BIO mean this token begins new mention
#I in BIO means this token continues previous mention of same label
#O in BIO means this token has no label
#this function take 2 list, first list contains token and correspoding
#index in second list contains tag
'''
example input:
Token     Tag
Apple     B-ORG
hired     O
John      B-PER
Smith     I-PER
in        O
Dublin    B-LOC
.         O

example output:
[
  ("ORG", "Apple"),
  ("PER", "John Smith"),
  ("LOC", "Dublin")
]

'''

def bio_to_mention_conv(tokens,tags):
    if len(tokens)!=len(tags):
        raise ValueError("tokens and tags must have the same length")
    output = []
    current_mention = []
    current_entity_type=""
    for token,tag in zip(tokens,tags):

        #at tag O, if a mention is currently open, save it to the output.
        if tag[0] == "O": 
            if current_mention:
                output.append((current_entity_type, " ".join(current_mention)))
                current_mention=[]
            current_entity_type=tag[0]
            continue
        
        if tag[0:2]=="B-":

            if current_mention:
                output.append((current_entity_type, " ".join(current_mention)))
                current_mention=[]
                
            current_mention.append(token)
            current_entity_type=tag[2:]
            continue

        if tag[0:2]=="I-":
            #if tag matches
            if current_entity_type==tag[2:]:
                current_mention.append(token)

            #for inavlid I tags, we will clsoe current mention and start new, and inform user
            else:
                current_mention=[]
                current_mention.append(token)
                current_entity_type=tag[2:]
                warnings.warn(f"Invalid BIO sequence: '{tag}' does not continue '{current_entity_type}'.\n Ignoring token '{token}'")
                

    if current_mention:  #for last tokens who may not be saved bcs there were no more tokens to continue loop
        output.append((current_entity_type, " ".join(current_mention)))


    return output

        
            
