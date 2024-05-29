import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Input sentence
sentence = "a cat is sitting in a toilet bowl in a bathroom with a towel on the floor and a toilet paper roll on the wall next to the toilet paper dispenser."

# Process the text
doc = nlp(sentence)

# Initialize a dictionary to hold the nouns, verbs, and adjectives
parts_of_speech = {
    "nouns": [],
    "adjectives": [],
    "verbs": []
}

# Extract nouns, adjectives, and verbs
for token in doc:
    if token.pos_ == 'NOUN':
        parts_of_speech["nouns"].append(token.text)
    elif token.pos_ == 'ADJ':
        parts_of_speech["adjectives"].append(token.text)
    elif token.pos_ == 'VERB':
        parts_of_speech["verbs"].append(token.text)

# Display the parts of speech
print("Parts of Speech Extracted:")
print(parts_of_speech)
