import os
import json
import spacy
from collections import defaultdict
import sys
# Load the NLP model just once
nlp = spacy.load("en_core_web_sm")

# Ensure the import path is correctly handled
sys.path.insert(0, '../')
from Question_type import All_task

# def extract_noun_phrases(sentence):
#     doc = nlp(sentence)
#     refined_noun_phrases = []
#     for chunk in doc.noun_chunks:
#         # Exclude initial determiner
#         first_token = chunk[0]
#         if first_token.dep_ == 'det':
#             refined_phrase = chunk.text[len(first_token.text)+1:]
#         else:
#             refined_phrase = chunk.text
#         if refined_phrase:
#             refined_noun_phrases.append(refined_phrase)
#     return refined_noun_phrases

# def extract_pos(sentence):
#     doc = nlp(sentence)
#     parts_of_speech = []
#     for token in doc:
#         if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
#             parts_of_speech.append(token.text)
#     return parts_of_speech

# if __name__ == "__main__":
#     task_idx = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))
#     task = Sg_task['function']['oarlks'][task_idx]
#     source_file = f'../datasets/npy_cap_all/function/fcl_mmf_{task}_train.json'
#     output_file = source_file.replace('.json', '_updated.json')
#     print(f"Loading file from {source_file}")
#     with open(source_file, 'r') as f:
#         data = json.load(f)
#     new_data = []
#     total = len(data)
#     for i, datum in enumerate(data):
#         caption = datum['caption'].split('.')[0] + '.'
#         entities = extract_noun_phrases(caption)
#         entities.extend(extract_pos(caption))
#         unique_entities = list(set(entities))
#         datum['entities'] = unique_entities
#         datum['caption'] = caption
#         new_data.append(datum)
#         if (i + 1) % 1000 == 0:
#             samples_done = i + 1
#             samples_left = total - samples_done
#             print(f'{samples_done} samples processed.')
#             print(f'Samples left: {samples_left}')
    
#     with open(output_file, 'w') as f:
#         json.dump(new_data, f, indent=4)
#     print(f'Storing new entities @ {output_file}')

def count():
    orig_path = '../datasets/vqa/Partition_Q_V2_no_ents/karpathy_train_q_subcategory.json'
    with open(orig_path, 'r') as f:
        data = json.load(f)
    final_data = []
    total = 0
    counts = {}
    for sub_task in All_task[:9]:
        counts[sub_task] = 0
        question_key = f'Q_{sub_task}'
        answer_key = f'A_q_{sub_task}'
        sub_data = [datum for datum in data if question_key in datum]
        
        for datum in sub_data:
            total +=1
            counts[sub_task] +=1
    print(total)
    print(counts)

def main():
    orig_path = '../datasets/vqa/Partition_Q_V2_subset_ST/karpathy_train_q_causal.json'
    with open(orig_path, 'r') as f:
        data = json.load(f)
    final_data = []
    total = 0
    for sub_task in All_task[:9]:
        question_key = f'Q_{sub_task}'
        answer_key = f'A_cap_{sub_task}'
        sub_data = [datum for datum in data if question_key in datum and answer_key in datum]
        new_data = []
        for datum in sub_data:
            new_datum = {key: value for key, value in datum.items() if key not in [question_key, answer_key]}
            question, answer = datum[question_key], datum[answer_key]
            new_datum['Q'] = question
            new_datum['A'] = answer.strip()
            new_data.append(new_datum)
            total+=1
        final_data.extend(new_data)
    print(f"Obtain causal data containing {total} samples")
    with open('../datasets/vqa/Partition_Q_V2_no_ents/karpathy_train_q_causalunbalanced_new.json', 'w') as f:
        json.dump(final_data, f)

if __name__ == "__main__":
    count()
