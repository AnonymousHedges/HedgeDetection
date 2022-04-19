# Attention-CNN

import torch
import numpy as np 
import pandas as pd
import optuna
import sklearn
from torch.optim import AdamW
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from vector_features_sentence import get_liwc, get_ngram, get_ngram_pos_tag
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from ast import literal_eval
from classifier_rule_based_no_syntax import classify_sentence, precision_rules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import argparse

MAX_EPOCHS = 10
BATCHSIZE = 32
CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 300
MAX_SENT_LEN = 64

class IndirectnessDataset(nn.Module):
    def __init__(self, word_embeddings, labels):
        self.word_embeddings = word_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.word_embeddings)

    def __getitem__(self, idx):
        # If the sentence was blank
        if len(self.word_embeddings[idx].shape) == 1:
            self.word_embeddings[idx] = np.zeros((MAX_SENT_LEN, EMBEDDING_SIZE))
        return torch.tensor(np.concatenate((np.zeros((max(0,MAX_SENT_LEN-len(self.word_embeddings[idx])), EMBEDDING_SIZE)), self.word_embeddings[idx])), dtype=torch.float), \
            torch.tensor(self.labels[idx], dtype=torch.long)

class AttentionCNN(nn.Module):
    def __init__(self, out_channels_conv, channel_width, dropout):
        super().__init__()
        self.out_channels_conv = out_channels_conv
        self.channel_width = channel_width
        self.dropout = dropout

        self.conv = nn.Conv2d(1, out_channels_conv, (EMBEDDING_SIZE, channel_width))
        self.dropout = nn.Dropout(dropout)
        self.pool_width = MAX_SENT_LEN - (channel_width-1)
        self.pool = nn.MaxPool2d((1, self.pool_width))
        self.fc = nn.Linear(self.out_channels_conv, CLASSES)

    def attention_net(self, x):
        """
        Use a trained attention net to perform dot-product attention on the kernels (Self-attention)
        If {ker_j} are the kernels, with k in [1, n], then we will compute a_j = Softmax(Sum_i=1->n(ker_j * ker_i)),
        and output {ker_j*} = a_j*{ker_j}

        Dot-product attention : Let say qj (=ker_j) is our query ; eij = qj*kij ; aj = Softmax(eij, Sum(eik)) ; oj = Sum(aj*ker_j)
        """

        # A batch_size * num_kernel * num_kernel that we are going to multiply with the transposed(batch_size * num_kernel) matrix
        # and sum to get the (batch_size * num_kernel * num_kernel) that we need to compute attention.
        x = x.squeeze(-1)
        kernel_matrix = x.repeat(1, 1, self.out_channels_conv)
        pre_attn = torch.matmul(kernel_matrix, x)
        attn = torch.sum(pre_attn, 2) 
        attn = torch.nn.functional.softmax(attn, 1)

        # In this step, we are going to multiply the (batch_size * num_kernel * num_kernel) with the (batch_size * num_kernel * size_kernel)
        # to get the (batch_size * num_kernel * size_kernel) we need
        output = torch.matmul(attn.unsqueeze(-1).repeat(1,1,attn.size()[1]), x).squeeze(-1)
        return output

    def forward(self, x):
        x = x.unsqueeze(-1).transpose(1,3)
        x = self.pool(F.relu(self.conv(x)))
        x = self.attention_net(x)
        x = self.dropout(self.fc(x.squeeze(-1).squeeze(-1)))
        x = torch.flatten(x,1)
        output = F.log_softmax(x, dim=1)
        return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', action="store", help = "Choose whether to evaluate the model ('eval') or to search for hyper-parameters ('hyper_opt')")
    args = parser.parse_args()

    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    dict_glove = None
    # Very costly in RAM, downgrade to 200d or smaller if it crashes.
    with open("./glove/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        dict_glove = {l[0]:np.array([float(x) for x in l[1:]]) for l in lines}

    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] if x.split()[0] != "IDE" else "IDQ" for x in list(dataset["Label"])]

    mapping_indirectness = {
        0 : "Nothing",
        1 : "IDA",
        2 : "IDS", 
        3 : "IDQ"
    }

    inverse_mapping_indirectness = {v:k for k,v in mapping_indirectness.items()}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 1)
    X = dataset.drop(columns=["Label"])
    Y = dataset["Label"]
    skf.get_n_splits(X, Y)

    confusion_matrices = []
    classification_reports = []
    f1_scores = []
    all_classes_f1_scores = []

    for train_indexes, valid_indexes in skf.split(X, Y):
        
        train_features, valid_features = X.iloc[train_indexes], X.iloc[valid_indexes]
        train_labels, valid_labels = list(Y.iloc[train_indexes]), list(Y.iloc[valid_indexes])

        train_texts = [literal_eval(x)[1] for x in train_features["Text"]]
        valid_texts = [literal_eval(x)[1] for x in valid_features["Text"]]

        # Add the Glove embeddings

        train_embeddings = [np.array([dict_glove[x] if x in dict_glove else np.zeros(EMBEDDING_SIZE) for x in word_tokenize(text)]) for text in train_texts]
        valid_embeddings = [np.array([dict_glove[x] if x in dict_glove else np.zeros(EMBEDDING_SIZE) for x in word_tokenize(text)]) for text in valid_texts]

        # Classes balancing through the loss

        class_weights = compute_class_weight("balanced", classes = np.unique(train_labels), y=train_labels)
        class_weights = [class_weights[-1], class_weights[0], class_weights[2], class_weights[1]]
        class_weights = np.sqrt(class_weights)

        if args.run_mode == "hyper_opt":
        
            def objective(trial, train_features, train_labels, valid_features, valid_labels, mapping_indirectness, class_weights):


                out_channels_conv = trial.suggest_int("out_channels_conv", 12, 16) 
                dropout = trial.suggest_float("dropout", 0.15, 0.25)
                channel_width = trial.suggest_int("channel_width", 1, 4)
                            
                train_labels = [inverse_mapping_indirectness[x] for x in train_labels]
                train_dataset = IndirectnessDataset(train_embeddings, train_labels)
                train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = False)

                valid_labels = [inverse_mapping_indirectness[x] for x in valid_labels]
                valid_dataset = IndirectnessDataset(valid_embeddings, valid_labels)
                valid_dataloader = DataLoader(valid_dataset, batch_size = BATCHSIZE, shuffle = False)
                
                model = AttentionCNN(out_channels_conv, channel_width, dropout)
                lr = trial.suggest_float("lr", 1e-5, 5e-3)
                loss_fct = nn.NLLLoss(weight = torch.tensor(class_weights, dtype=torch.float))
                optimizer = AdamW(model.parameters(), lr=lr)

                prediction_list = []

                best_accuracy = 0

                # Training of the model.
                for epoch in range(MAX_EPOCHS):
                    model.train()
                    for batch_idx, (data, target) in enumerate(train_dataloader):
                        data, target = data.to(DEVICE), target.to(DEVICE)

                        optimizer.zero_grad()
                        output = model(data)
                        loss = loss_fct(output, target)
                        loss.backward()
                        optimizer.step()

                    # Validation of the model.
                    model.eval()
                    correct = 0
                    prediction_list = []
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(valid_dataloader):
                            
                            data, target = data.to(DEVICE), target.to(DEVICE)
                            output = model(data)
                            # Get the index of the max log-probability.
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                            prediction_list.append(pred.detach().numpy())

                    accuracy = correct / len(valid_dataloader.dataset)
                    if accuracy < best_accuracy:
                        break
                    else:
                        best_accuracy = accuracy

                prediction_list = np.concatenate(prediction_list).transpose(1,0).flatten()
                pred_labels = [mapping_indirectness[x] for x in prediction_list]
                valid_labels = [mapping_indirectness[x] for x in valid_labels]
                f1_score = sklearn.metrics.f1_score(valid_labels, pred_labels,
                    labels = [x for x in ["IDA", "IDS", "IDQ"]],
                    average="weighted")
                
                print(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))

                return f1_score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial:objective(trial, train_features, train_labels, valid_features, valid_labels, mapping_indirectness, class_weights), n_trials=100)
            print(study.best_trial)

        elif args.run_mode == "eval":

            # Train after hyper-parameter search

            train_labels = [inverse_mapping_indirectness[x] for x in train_labels]
            train_dataset = IndirectnessDataset(train_embeddings, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = False)

            valid_labels = [inverse_mapping_indirectness[x] for x in valid_labels]
            valid_dataset = IndirectnessDataset(valid_embeddings, valid_labels)
            valid_dataloader = DataLoader(valid_dataset, batch_size = BATCHSIZE, shuffle = False)
            
            model = AttentionCNN(14, 3, 0.202)
            lr = 0.001
            loss_fct = nn.NLLLoss(weight = torch.tensor(class_weights, dtype=torch.float))
            optimizer = AdamW(model.parameters(), lr=lr)

            prediction_list = []

            best_accuracy = 0

            # Training of the model.
            for epoch in range(MAX_EPOCHS):
                model.train()
                for batch_idx, (data, target) in enumerate(train_dataloader):
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fct(output, target)
                    loss.backward()
                    optimizer.step()

                # Validation of the model.
                model.eval()
                correct = 0
                prediction_list = []
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(valid_dataloader):
                        
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        output = model(data)
                        # Get the index of the max log-probability.
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        prediction_list.append(pred.detach().numpy())

                accuracy = correct / len(valid_dataloader.dataset)
                if accuracy < best_accuracy:
                    break
                else:
                    best_accuracy = accuracy

            prediction_list = np.concatenate(prediction_list).transpose(1,0).flatten()
            pred_labels = [mapping_indirectness[x] for x in prediction_list]
            valid_labels = [mapping_indirectness[x] for x in valid_labels]
            f1_score = sklearn.metrics.f1_score(valid_labels, pred_labels,
                labels = [x for x in ["IDA", "IDS", "IDQ"]],
                average="weighted")
            
            all_classes_f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels, average="weighted"))
            f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels,
                labels = ["IDA", "IDS", "IDQ"],
                average="weighted"))
            
            classification_reports.append(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))
            confusion_matrices.append(confusion_matrix(valid_labels, pred_labels))

    if args.run_mode=="eval":    
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

    