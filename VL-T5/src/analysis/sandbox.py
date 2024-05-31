import spacy
nlp = spacy.load("en_core_web_sm")

import spacy

def extract_noun_phrases(sentence):
    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the sentence
    doc = nlp(sentence)
    
    # List to hold the refined noun phrases
    refined_noun_phrases = []

    # Extract noun phrases and remove leading determiners
    for chunk in doc.noun_chunks:
        # Split the chunk into tokens and filter out 'a', 'an', 'the' if they are the first token
        tokens = [token for token in chunk if token.dep_ != 'det' or token.i > chunk.start]
        refined_phrase = ' '.join(token.text for token in tokens)
        if refined_phrase:  # Ensure the phrase is not empty
            refined_noun_phrases.append(refined_phrase)
    
    return refined_noun_phrases


def extract_pos(sentence):
    # Load the English NLP model
    
    
    # Process the sentence
    doc = nlp(sentence)
    
    # Dictionary to hold the parts of speech
    parts_of_speech = []
    
    # Extract nouns, adjectives, and verbs
    for token in doc:
        if token.pos_ == 'ADJ':
            parts_of_speech.append(token.text)
        elif token.pos_ == 'VERB':
            parts_of_speech.append(token.text)
    
    return parts_of_speech


if __name__ == "__main__":
    task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
    task = Sg_task['function']['oarlks'][task_idx]
    source_file  = f'../datasets/npy_cap_all/fcl_mmf_{task}_train.json'
    with open(source_file, 'r') as f:
        data = json.load(f)
    data=data[:50]
    new_data = []
    for datum in data:
        caption = datum['caption'].split('.')[0]+ '.'
        entities = extract_noun_phrases(sentence)
        entities.extend(extract_pos(sentence))
        datum['entities'] = entities
        new_data.append(datum)

    with open(source_file, 'w') as f:
        json.dump(d, new_data, indent=4)
