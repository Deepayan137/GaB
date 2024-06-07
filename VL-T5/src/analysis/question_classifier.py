import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
from tqdm import *

from torch.multiprocessing import set_start_method

# Try setting the start method to 'spawn'
try:
    set_start_method('spawn')
except RuntimeError:
    pass  


cat_dict = {
    "object":
            {"{'detailed': 'place', 'semantic': 'global', 'structural': 'query'}":0,
            "{'detailed': 'category', 'semantic': 'cat', 'structural': 'query'}":1,
            "{'detailed': 'objThisChoose', 'semantic': 'cat', 'structural': 'choose'}":2,
            "{'detailed': 'categoryThisChoose', 'semantic': 'cat', 'structural': 'choose'}":3,
            "{'detailed': 'categoryThis', 'semantic': 'cat', 'structural': 'query'}":4,
            "{'detailed': 'placeChoose', 'semantic': 'global', 'structural': 'choose'}":5,
    },
    "attribute":
            {"{'detailed': 'chooseAttr', 'semantic': 'attr', 'structural': 'choose'}":0,
            "{'detailed': 'categoryThat', 'semantic': 'cat', 'structural': 'query'}":1,
            "{'detailed': 'categoryAttr', 'semantic': 'cat', 'structural': 'query'}":2,
            "{'detailed': 'categoryThatChoose', 'semantic': 'cat', 'structural': 'choose'}":3,
            "{'detailed': 'directWhich', 'semantic': 'attr', 'structural': 'query'}":4,
            "{'detailed': 'activity', 'semantic': 'attr', 'structural': 'query'}":5,
            "{'detailed': 'activityWho', 'semantic': 'cat', 'structural': 'query'}":6,
            "{'detailed': 'material', 'semantic': 'attr', 'structural': 'query'}":7,
            "{'detailed': 'directOf', 'semantic': 'attr', 'structural': 'query'}":8,
            "{'detailed': 'weather', 'semantic': 'global', 'structural': 'query'}":9,
            "{'detailed': 'how', 'semantic': 'attr', 'structural': 'query'}":10,
            "{'detailed': 'locationChoose', 'semantic': 'global', 'structural': 'choose'}":11,
            "{'detailed': 'materialChoose', 'semantic': 'attr', 'structural': 'choose'}":12,
            "{'detailed': 'typeChoose', 'semantic': 'attr', 'structural': 'choose'}":13,
            "{'detailed': 'activityChoose', 'semantic': 'attr', 'structural': 'choose'}":14,
            "{'detailed': 'company', 'semantic': 'attr', 'structural': 'query'}":15,
            "{'detailed': 'weatherChoose', 'semantic': 'global', 'structural': 'choose'}":16,
            "{'detailed': 'state', 'semantic': 'attr', 'structural': 'query'}":17,
            "{'detailed': 'companyChoose', 'semantic': 'attr', 'structural': 'choose'}":18,
            "{'detailed': 'stateChoose', 'semantic': 'attr', 'structural': 'choose'}":19
    },
    "relation":{
            "{'detailed': 'relO', 'semantic': 'rel', 'structural': 'query'}":0,
            "{'detailed': 'relS', 'semantic': 'rel', 'structural': 'query'}":1,
            "{'detailed': 'directOf', 'semantic': 'attr', 'structural': 'query'}":2,
            "{'detailed': 'how', 'semantic': 'attr', 'structural': 'query'}":3,
            "{'detailed': 'chooseAttr', 'semantic': 'attr', 'structural': 'choose'}":4,
            "{'detailed': 'categoryRelS', 'semantic': 'rel', 'structural': 'query'}":5,
            "{'detailed': 'directWhich', 'semantic': 'attr', 'structural': 'query'}":6,
            "{'detailed': 'relVerify', 'semantic': 'rel', 'structural': 'verify'}":7,
            "{'detailed': 'activity', 'semantic': 'attr', 'structural': 'query'}":8,
            "{'detailed': 'materialChoose', 'semantic': 'attr', 'structural': 'choose'}":9,
            "{'detailed': 'material', 'semantic': 'attr', 'structural': 'query'}":10,
            "{'detailed': 'categoryRelO', 'semantic': 'rel', 'structural': 'query'}":11,
            "{'detailed': 'relVerifyCr', 'semantic': 'rel', 'structural': 'verify'}":12,
            "{'detailed': 'relChooser', 'semantic': 'rel', 'structural': 'choose'}":13,
            "{'detailed': 'relVerifyCo', 'semantic': 'rel', 'structural': 'verify'}":14,
            "{'detailed': 'activityChoose', 'semantic': 'attr', 'structural': 'choose'}":15,
            "{'detailed': 'sameRelate', 'semantic': 'rel', 'structural': 'query'}":16,
            "{'detailed': 'categoryRelOChoose', 'semantic': 'rel', 'structural': 'choose'}":17,
            "{'detailed': 'dir', 'semantic': 'rel', 'structural': 'query'}":18,
            "{'detailed': 'sameMaterialRelate', 'semantic': 'rel', 'structural': 'query'}":19,
            "{'detailed': 'positionChoose', 'semantic': 'attr', 'structural': 'choose'}":20,
            "{'detailed': 'existRelS', 'semantic': 'rel', 'structural': 'verify'}":21,
            "{'detailed': 'company', 'semantic': 'attr', 'structural': 'query'}":22
    },
    "logical":{
            "{'detailed': 'twoDifferentC', 'semantic': 'attr', 'structural': 'compare'}":0,
            "{'detailed': 'twoSame', 'semantic': 'attr', 'structural': 'compare'}":1,
            "{'detailed': 'twoCommon', 'semantic': 'attr', 'structural': 'compare'}":2,
            "{'detailed': 'twoDifferent', 'semantic': 'attr', 'structural': 'compare'}":3,
            "{'detailed': 'diffAnimalsC', 'semantic': 'attr', 'structural': 'compare'}":4,
            "{'detailed': 'sameAnimals', 'semantic': 'attr', 'structural': 'compare'}":5,
            "{'detailed': 'twoSameC', 'semantic': 'attr', 'structural': 'compare'}":6,
            "{'detailed': 'sameAnimalsC', 'semantic': 'attr', 'structural': 'compare'}":7,
            "{'detailed': 'twoSameMaterialC', 'semantic': 'attr', 'structural': 'compare'}":8,
            "{'detailed': 'twoSameMaterial', 'semantic': 'attr', 'structural': 'compare'}":9,
            "{'detailed': 'comparativeChoose', 'semantic': 'attr', 'structural': 'compare'}":10,
            "{'detailed': 'sameGenderC', 'semantic': 'attr', 'structural': 'compare'}":11,
            "{'detailed': 'sameGender', 'semantic': 'attr', 'structural': 'compare'}":12,
            "{'detailed': 'diffAnimals', 'semantic': 'attr', 'structural': 'compare'}":13,
            "{'detailed': 'diffGender', 'semantic': 'attr', 'structural': 'compare'}":14
    },
    "knowledge":{},
}


# Load the model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using:{device}")
def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get the sentence embedding."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

def get_embedding(sentence):
    """Generate a normalized embedding for a single sentence."""
    # Tokenize the single sentence
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Mean pooling to get the sentence embedding
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize the embedding
    normalized_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    return normalized_embedding


class QuestionTypeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuestionTypeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



class QtypeDataset(Dataset):
    def __init__(self, task='object', split='train', scenario='function',):
        super().__init__()
        filename = f'fcl_mmf_{task}_{split}.npy'
        data_path = os.path.join('../datasets/npy', scenario, filename)
        self.data = np.load(data_path, allow_pickle=True)
        self.task = task

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        question  = datum['question']
        question_type = datum["raw_question_type"]
        label = cat_dict[self.task][f'{question_type}']
        embedding = get_embedding(question)
        return {'embedding':embedding, "label":label, 'question':question}

def get_dataloader(task, split, scenario='function', batch_size=32):
    dataset = QtypeDataset(task=task, split=split, scenario=scenario)
    dataloader = DataLoader(dataset, 
        batch_size=batch_size, shuffle=True if split=='train' else False, num_workers=0)
    return dataloader

def evaluate_model(classifier, val_loader):
    classifier.eval()

    # Initialize counters
    correct_predictions = 0
    total_predictions = 0

    # No gradient is needed
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            # Predict the outputs
            outputs = classifier(embeddings.squeeze())
            _, predicted = torch.max(outputs, 1)

            # Update the counters
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    # Calculate the accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def main():
    # Parameters
    task='object'
    input_dim = 768  # Adjust based on your embeddings
    hidden_dim = 256 # You can tune this
    output_dim = len(cat_dict[task].keys())  # Adjust based on the number of question types
    
    # Initialize model, loss, and optimizer
    print(f"Building a classifier for task {task} with {output_dim} classes")
    classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
    train_loader = get_dataloader(task=task, split='train')
    val_loader = get_dataloader(task=task, split='val')
    # if os.path.exists('ckpt/best_model.pth'):
    #     print("Loading existsing checkpoint")
    #     ckpt = torch.load('ckpt/best_model.pth')
    #     classifier.load_state_dict(ckpt)
    # accuracy=evaluate_model(classifier, val_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)
    
    best = 0.0
    patience_counter = 0
    patience = 3
    for epoch in range(0, 20):
        classifier.train()
        for batch in tqdm(train_loader):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            outputs = classifier(embeddings.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy=evaluate_model(classifier, val_loader)
        print(f"Accuracy after {epoch+1}: {accuracy*100.}")
        if accuracy > best:
            best=accuracy
            patience_counter = 0
            torch.save(classifier.state_dict(), f'ckpt/{task}.pth')
            print(f"New best model saved with accuracy: {best:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
        if patience_counter > patience:
            print("Early stopping triggered.")
            print("Saving Last")
            break  # Break out of the training loop

if __name__ == "__main__":
   main()

