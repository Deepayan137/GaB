import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The quick brown fox jumps over the lazy dog who was sleeping on the porch."

# Process the text with spaCy
doc = nlp(text)

# Extract noun phrases
noun_phrases = [chunk.text for chunk in doc.noun_chunks]

# Print the noun phrases
print("Noun Phrases:", noun_phrases)

# Detailed view: print each noun phrase with its root text, root dependency, and root head text
for chunk in doc.noun_chunks:
    print(f"Noun phrase: '{chunk.text}' | Root text: '{chunk.root.text}', Root dep: '{chunk.root.dep_}', Root head text: '{chunk.root.head.text}'")