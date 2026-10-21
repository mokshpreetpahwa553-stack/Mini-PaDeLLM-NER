# while training we give input like
#Input:
# 1. PER count
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: PER
# How many PER entities are in the sentence?
#
# Target:
# 1

# 2. PER mention 1
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: PER
# What is the 1st PER entity?
#
# Target:
# John Smith

# 3. ORG count
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: ORG
# How many ORG entities are in the sentence?
#
# Target:
# 1

# 4. ORG mention 1
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: ORG
# What is the 1st ORG entity?
#
# Target:
# Apple

# 5. LOC count
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: LOC
# How many LOC entities are in the sentence?
#
# Target:
# 1

# 6. LOC mention 1
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: LOC
# What is the 1st LOC entity?
#
# Target:
# Dublin

# 7. MISC count
# Input:
# Sentence: Apple hired John Smith in Dublin .
# Entity type: MISC
# How many MISC entities are in the sentence?
#
# Target:
# 0

#SO ONE SENTENCE SPLIT INTO 7 TRAINING EXAMPLES
#NOTE: DURING TRAINING, LOSS IS COMPUTED ON TARGET TOKENS NOT ON PROMPT TOKENS


#FUNTION TO BUILD TRAING EXAMPLES
FUNCTION build_training_examples(text, entity_label , all_labels): #entity label is list containing pairs of entity and its label like ("john","PER") , all_labels is list of all labels
	
	training_examples=[]
	
	for label in all_labels:
		entities = list of all entites with this label  # it will contain pair for each label with entity9
		total_entities = len(entities)
		
		
		prompt_to_count = count_prompt(text,label)
		count_target = total_entities
		
		ADD (prompt_to_count , count_target) to training_example
		
		for n from 1 to total_entities: #if mention_count == 0 , it wont do anyhting
			
			prompt_to_find_mention = mention_prompt(text, label , n)
			mention_target = entities[n-1][0]

			ADD (prompt_to_find_mention , mention_target) TO training_examples
	RETURN training_examples 
			

#FUNTION TO PREDICT NAMED ENTITES PARALLELY LIKE PaDeLLM-NER
FUNCTION predict_entities(text, all_labels, model): #model here is LLM model using
	
	count_predictions={}
	for label in all_labels:
		prompt_to_predict_count = count_prompt(text,label)
		count, probability =  model.predict_count(prompt_to_predict_count)
		count_predictions[label] = count

	entity_prediction_requests=[]
	for label in all_labels:
		for n from 1 to count_predictions[label]:
			prompt_to_predict_nth_entity = entity_prompt(text,label,n)
			ADD (label,n,prompt_to_predict_nth_entity ) to entity_prediction_requests

	predicted_entities = model.batch_predict_entity(entity_prediction_requests)
	
	candidates=[]
	for each_prediction in predicted_entities:
		#we may need ot change indexes as it is just arbitary
		entity,label,probability = each_prediction[0],each_prediciton[1],each_prediciton[2]
		
		ADD (entity,label,probability) to candidates
	
	final_entities = remove_duplicates(candidates)
	RETURN final_entities
	
#FUNCTION TO REMOVE DUPLICATES
FUNCTION remove_duplicates(list_of_candidates): #candidate is a list containig (entity, label, probability)
	
	final_candidates=[]
	groups_of_entities = GROUP all candidates by normalised mention text(apple same as Apple)
	
	for each group in groups_of_entities:
		final = candidate with highest probability
		ADD final TO final_candidates
	
	RETURN final_candidates
	
	
	



