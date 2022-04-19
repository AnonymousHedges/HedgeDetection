# BERT fine-tuning

import torch
import numpy as np 
import pandas as pd
import sklearn
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from ast import literal_eval

MAX_EPOCHS = 3
BATCHSIZE = 16
CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertIndirectnessDataset(torch.nn.Module):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

if __name__ == "__main__":

    # Load the dataset

    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] if x.split()[0] != "IDE" else "IDQ" for x in list(dataset["Label"])]

    mapping_indirectness = {
        0 : "Nothing",
        1 : "IDA",
        2 : "IDS", 
        3 : "IDQ"
    }

    inverse_mapping_indirectness = {v:k for k,v in mapping_indirectness.items()}
    
    # Cut dataset in sub-parts 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    X = dataset.drop(columns=["Label"])
    Y = dataset["Label"]
    skf.get_n_splits(X, Y)

    confusion_matrices = []
    classification_reports = []
    # F1 scores with 3 classes
    f1_scores = [] 
    # F1 scores with 4 classes
    all_classes_f1_scores = []

    for train_indexes, valid_indexes in skf.split(X, Y):
        
        train_features, valid_features = X.iloc[train_indexes], X.iloc[valid_indexes]
        train_labels, valid_labels = list(Y.iloc[train_indexes]), list(Y.iloc[valid_indexes])

        train_texts = [literal_eval(x)[1] for x in train_features["Text"]]
        valid_texts = [literal_eval(x)[1] for x in valid_features["Text"]]

        # Add the BERT encodings

        train_encodings = tokenizer(train_texts, padding="max_length", truncation=True)
        valid_encodings = tokenizer(valid_texts, padding="max_length", truncation=True)

        # Balancing classes

        class_weights = compute_class_weight("balanced", classes = np.unique(train_labels), y=train_labels)
        class_weights = [class_weights[-1], class_weights[0], class_weights[2], class_weights[1]]
        class_weights = np.sqrt(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float, device = DEVICE)

        # Construct the training and validation loaders

        train_labels = [inverse_mapping_indirectness[x] for x in train_labels]
        train_dataset = BertIndirectnessDataset(train_encodings, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = False)

        valid_labels = [inverse_mapping_indirectness[x] for x in valid_labels]
        valid_dataset = BertIndirectnessDataset(valid_encodings, valid_labels)
        valid_dataloader = DataLoader(valid_dataset, batch_size = BATCHSIZE, shuffle = False)
        
        # Construct the model 

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 4)
        model.to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        loss_fct = nn.CrossEntropyLoss(weight = class_weights)
        prediction_list = []
        best_accuracy = 0

        # Training of the model.
        for epoch in range(MAX_EPOCHS):
            model.train()
            total_duration = len(train_dataloader)
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fct(output[1], labels)
                loss.backward()
                optimizer.step()
                print(i/total_duration)

            # Validation of the model.
            model.eval()
            correct = 0
            prediction_list = []
            with torch.no_grad():
                for batch in valid_dataloader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    pred = output[1].argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels).sum().item()
                    prediction_list.append(pred.detach().cpu().numpy())

            accuracy = correct / len(valid_dataloader.dataset)
            if accuracy < best_accuracy:
                break
            else:
                best_accuracy = accuracy

        prediction_list = np.concatenate(prediction_list).transpose(1,0).flatten()
        pred_labels = [mapping_indirectness[x] for x in prediction_list]
        valid_labels = [mapping_indirectness[x] for x in valid_labels]
        
        all_classes_f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels, average="weighted"))
        f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels,
            labels = ["IDA", "IDS", "IDQ"],
            average="weighted"))
        
        classification_reports.append(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))
        confusion_matrices.append(confusion_matrix(valid_labels, pred_labels))
    
    print("Mean F1_score:"+str(np.mean(f1_scores)))
    print("Std F1_score:"+str(np.std(f1_scores)))
    print("Mean all F1_score:"+str(np.mean(all_classes_f1_scores)))
    print("Std all F1_score:"+str(np.std(all_classes_f1_scores)))
    print("Classification reports : \n")
    for x in classification_reports:
        print(x)
        print()
    print("Confusion matrices : \n")    
    for x in confusion_matrices:
        print(x)
        print()
