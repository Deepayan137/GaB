import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import os
from tqdm import *
import sys

sys.path.insert(0, "../")
from Question_type import *
from torch.multiprocessing import set_start_method

# Try setting the start method to 'spawn'
try:
    set_start_method("spawn")
except RuntimeError:
    pass

cat_dict = {
    "q_recognition": {
        "what": 0,
        "what is the": 1,
        "what is": 2,
        "what are the": 3,  # Consolidated duplicate here
        "which": 4,
        "what is on the": 5,
        "what is in the": 6,
        "what is this": 7,
        "what does the": 8,
        "who is": 9,
        "what is the name": 10,
        "what are": 11,
    },
    "q_location": {"where is the": 0, "where are the": 1, "what room is": 2},
    "q_judge": {
        "is the": 0,
        "is this": 1,
        "is this a": 2,
        "are the": 3,
        "is there a": 4,
        "is it": 5,
        "is there": 6,
        "is": 7,
        "are there": 8,
        "are these": 9,
        "are": 10,
        "are there any": 11,
        "is this an": 12,
        "was": 13,
        "is that a": 14,
    },
    "q_commonsense": {"does the": 0, "does this": 1, "do": 2, "has": 3, "do you": 4, "can you": 5, "could": 6},
    "q_count": {
        "how many": 0,
        "how many people": 1,
        "how many people are in": 2,
        "what number is": 3,
        "how many people are": 4,
    },
    "q_action": {
        "what is the man": 0,
        "is the man": 1,
        "are they": 2,
        "is he": 3,
        "is the woman": 4,
        "what is the person": 5,
        "what is the woman": 6,
        "is this person": 7,
        "is the person": 8,
        "is this": 9,
        "are the": 10,
        "are these": 11,
        "are": 12,
        "is": 13,
        "are there": 14,
        "is there": 15,
        "is this a": 16,
        "is the": 17,
        "is it": 18,
    },
    "q_color": {
        "what color is the": 0,
        "what color are the": 1,
        "what color": 2,
        "what color is": 3,
        "what is the color of the": 4,
    },
    "q_type": {"what kind of": 0, "what type of": 1},
    "q_subcategory": {
        "none of the above": 0,
        "what time": 1,
        "what sport is": 2,
        "what animal is": 3,
        "what brand": 4,
    },
}


# Load the model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Mean pooling to get the sentence embedding
    sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
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
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Adds a batch dimension
        x = self.softmax(x)
        return x


class QtypeDataset(Dataset):
    def __init__(
        self,
        task="q_recognition",
        split="train",
        scenario="function",
    ):
        super().__init__()
        data_info_path = os.path.join(f"../datasets/vqa/Partition_Q_V2/karpathy_{split}_" + f"{task}.json")
        with open(data_info_path, "r") as f:
            self.data = json.load(f)
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        if "sent" in datum:
            sent = datum["sent"]
        elif "question" in datum:
            sent = datum["question"]
        question_type = datum["question_type"]
        label = cat_dict[self.task][question_type]
        embedding = get_embedding(sent)
        return {"embedding": embedding, "label": label, "question": sent}


def get_dataloader(task, split, scenario="function", batch_size=32):
    dataset = QtypeDataset(task=task, split=split, scenario=scenario)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True if split == "train" else False, num_workers=0)
    return dataloader


def evaluate_model(classifier, val_loader):
    classifier.eval()

    # Initialize counters
    correct_predictions = 0
    total_predictions = 0

    # No gradient is needed
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)

            # Predict the outputs
            outputs = classifier(embeddings.squeeze())
            try:
                _, predicted = torch.max(outputs, 1)
            except:
                import pdb

                pdb.set_trace()
            # Update the counters
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    # Calculate the accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


def main():
    # Parameters
    task_idx = int(os.getenv("SLURM_ARRAY_TASK_ID", 5))
    # task='q_location'
    task = All_task[task_idx]
    input_dim = 768  # Adjust based on your embeddings
    hidden_dim = 256  # You can tune this
    output_dim = len(cat_dict[task].keys())  # Adjust based on the number of question types

    # Initialize model, loss, and optimizer
    print(f"Building a classifier for task {task} with {output_dim} classes")
    classifier = QuestionTypeClassifier(input_dim, hidden_dim, output_dim).to(device)
    train_loader = get_dataloader(task=task, split="train")
    val_loader = get_dataloader(task=task, split="val")
    # if os.path.exists('ckpt/best_model.pth'):
    #     print("Loading existsing checkpoint")
    #     ckpt = torch.load('ckpt/best_model.pth')
    #     classifier.load_state_dict(ckpt)
    # accuracy=evaluate_model(classifier, val_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

    best = 0.0
    for epoch in range(0, 3):
        classifier.train()
        for batch in tqdm(train_loader):
            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            outputs = classifier(embeddings.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy = evaluate_model(classifier, val_loader)
        print(f"Accuracy after {epoch+1}: {accuracy*100.}")
        if accuracy > best:
            best = accuracy
            torch.save(classifier.state_dict(), f"ckpt_vqacl/{task}.pth")
            print(f"New best model saved with accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
