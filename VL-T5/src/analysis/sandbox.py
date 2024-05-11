import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

text = "The photo is a child playing."

# Process the text
doc = nlp(text)

# Iterate over the recognized entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Type: {ent.label_}")

for token in doc:
    print(f"Token: {token.text}, POS: {token.pos_}, Detailed POS: {token.tag_}")

# Additional logic to classify as animal, person, or thing
# for ent in doc.ents:
#     if ent.label_ == "PERSON":
#         print(f"{ent.text} is a person.")
#     elif ent.label_ == "ANIMAL":
#         print(f"{ent.text} is an animal.")
#     elif ent.label_ == "ORG" or ent.label_ == "GPE" or ent.label_ == "PRODUCT":
#         print(f"{ent.text} is a thing (or organization/place/product).")
#     else:
#         print(f"{ent.text} is of type {ent.label_}, which may not specifically be a person, animal, or thing.")
