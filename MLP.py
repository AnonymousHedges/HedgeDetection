
import torch
import numpy as np 
import pandas as pd
import optuna
import sklearn
import argparse
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from vector_features_sentence import get_liwc, get_ngram, get_ngram_pos_tag
from ast import literal_eval
from classifier_rule_based_no_syntax import classify_sentence, precision_rules
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


MAX_EPOCHS = 100
BATCHSIZE = 32
CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndirectnessDataset(torch.nn.Module):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype =torch.long)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", help = "Choose the model representation, between Pretrained-Embeddings (pte), \
    Knowledge-Driven Features (kdf), or the combination of both (pte+kdf)")
    parser.add_argument('--run_mode', action="store", help = "Choose whether to evaluate the model ('eval') \
         or to search for hyper-parameters ('hyper_opt')")
    args = parser.parse_args()

    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] for x in list(dataset["Label"])]

    if args.model in ["pte", "pte+kdf"]:
        model = SentenceTransformer('all-mpnet-base-v2')
        dataset["Embeddings"] = [x for x in model.encode([literal_eval(x)[1] for x in list(dataset["Text"])])]

    mapping_indirectness = {
        0 : "Nothing",
        1 : "IDA",
        2 : "IDS", 
        3 : "IDQ"
    }

    inverse_mapping_indirectness = {v:k for k,v in mapping_indirectness.items()}
    
    columns_nonverbal_behaviors = ['Back_Channel_Tutor',
       'Back_Channel_Tutee', 'Gaze_Worksheet_Tutor', 'Smile_Tutor',
       'Head_Nod_Tutor', 'Gaze_Worksheet_Tutee', 'Gaze_Elsewhere_Tutee',
       'Gaze_Partner_Tutee', 'Head_Nod_Tutee', 'Gaze_Partner_Tutor',
       'Gaze_Elsewhere_Tutor', 'Smile_Tutee']

    columns_tutoring_moves = ['Incorrect_Feedback_Tutor',
       'Deep_Question_Tutee', 'Incorrect_Feedback_Tutee',
       'Knowledge_telling_Tutor', 'Metacognition_Tutor',
       'Knowledge_building_Tutor', 'Knowledge_building_Tutee',
       'Shallow_Question_Tutor', 'Shallow_Question_Tutee',
       'Correct_Feedback_Tutor', 'Correct_Feedback_Tutee',
       'Metacognition_Tutee', 'Knowledge_telling_Tutee',
       'Deep_Question_Tutor']

    mapping_tutoring_moves = {
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (0, 0, 0, 0, 0, 1),
        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) : (1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0) : (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0) : (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0) : (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0) : (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0) : (1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) : (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0) : (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0) : (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1) : (0, 0, 0, 0, 1, 0)
    }

    string_mapping_tutoring_moves = {"Feedback_tutor" : ['Incorrect_Feedback_Tutor', 'Correct_Feedback_Tutor', 'Metacognition_Tutor'],
     "Feedback_tutee" : ['Incorrect_Feedback_Tutee', 'Correct_Feedback_Tutee', 'Metacognition_Tutee'],
     "Affirmation_tutor" : ["Knowledge_building_Tutor", "Knowledge_telling_Tutor"],
     "Trial_tutee" : ["Knowledge_building_Tutee", "Knowledge_telling_Tutee"],
     "Question_tutor" : ["Deep_Question_Tutor", "Shallow_Question_Tutor"],
     "Question_tutee" : ["Deep_Question_Tutee", "Shallow_Question_Tutee"]}

    # Have to litteral eval the tuples
    dataset["Tutoring Moves"] = dataset["Tutoring Moves"].apply(literal_eval)
    dataset["Paraverbal Behaviors"] = dataset["Paraverbal Behaviors"].apply(literal_eval)
    dataset["Nonverbal Behaviors"] = dataset["Nonverbal Behaviors"].apply(literal_eval)

    # Replace the columns with the mapped values

    new_values = [mapping_tutoring_moves[x] if x in mapping_tutoring_moves else (0, 0, 0, 0, 0, 0) for x in dataset["Tutoring Moves"]] 
    dataset.drop(columns=["Tutoring Moves"], axis = 1, inplace = True)
    dataset["Tutoring Moves"] = new_values

    # Cut dataset in sub-parts 

    skf = StratifiedKFold(n_splits=5, shuffle=True)
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

        # Size clauses

        vectors_size = [len(word_tokenize(x)) for x in train_texts]
        valid_vectors_size = [len(word_tokenize(x)) for x in valid_texts]

        if args.model in ["kdf", "pte+kdf"]:

            # Ngrams

            train_dict_ngram, vectors_ngram = get_ngram(train_texts, train=True)
            valid_vectors_ngram = get_ngram(valid_texts, train=False, runtime_dict=train_dict_ngram)

            # Pos-tag

            train_dict_pos, vectors_ngram_pos = get_ngram_pos_tag(train_texts, train=True)
            valid_vectors_ngram_pos = get_ngram_pos_tag(valid_texts, train=False, runtime_dict=train_dict_pos)

            # LIWC 

            vectors_liwc = [get_liwc(x) for x in train_texts]
            valid_vectors_liwc = [get_liwc(x) for x in valid_texts]

            # Tutoring moves 

            vectors_tutoring_moves = list(train_features["Tutoring Moves"])
            valid_vectors_tutoring_moves = list(valid_features["Tutoring Moves"])

            # Paraverbal behaviors

            vectors_pvb = list(train_features["Paraverbal Behaviors"])
            valid_vectors_pvb = list(valid_features["Paraverbal Behaviors"])

            # Non-verbal behaviors 

            vectors_nvb = list(train_features["Nonverbal Behaviors"])
            valid_vectors_nvb = list(valid_features["Nonverbal Behaviors"])

            # Construct of features

            dict_precision = precision_rules(train_texts, train_labels)
            dict_precision["Nothing"] = {"0":1.0}
            vectors_precision = [classify_sentence(sent) for sent in train_texts]
            vectors_precision = [inverse_mapping_indirectness[x[0]]*dict_precision[x[0]][x[1]] for x in vectors_precision]
            valid_vectors_precision = [classify_sentence(sent) for sent in valid_texts]
            valid_vectors_precision = [inverse_mapping_indirectness[x[0]]*dict_precision[x[0]][x[1]] for x in valid_vectors_precision]

            train_kdf_features = [np.concatenate((
                # Clause length 
                np.array([v_size]),
                # ngrams
                np.array(v_ngram)/v_size,
                # Pos
                np.array(v_ngram_pos)/v_size,
                # Liwc
                np.array(v_liwc)/v_size,
                # Tutoring moves
                np.array(v_tutoring_moves),
                # Paraverbal behaviors
                np.array(v_pvb),
                # Nonverbal Behaviors
                np.array(v_nvb),
                # Precision
                np.array([v_precision])
            )) for v_size, v_ngram, v_ngram_pos, v_liwc, v_tutoring_moves, v_pvb, v_nvb, v_precision in 
            zip(vectors_size, vectors_ngram, vectors_ngram_pos, vectors_liwc, vectors_tutoring_moves, vectors_pvb, vectors_nvb, vectors_precision)]

            valid_kdf_features = [np.concatenate((
                # Clause length 
                np.array([v_size]),
                # ngrams
                np.array(v_ngram)/v_size,
                # Pos
                np.array(v_ngram_pos)/v_size,
                # Liwc
                np.array(v_liwc)/v_size,
                # Tutoring moves
                np.array(v_tutoring_moves),
                # Paraverbal behaviors
                np.array(v_pvb),
                # Nonverbal Behaviors
                np.array(v_nvb),
                # Precision
                np.array([v_precision])
            )) for v_size, v_ngram, v_ngram_pos, v_liwc, v_tutoring_moves, v_pvb, v_nvb, v_precision in 
            zip(valid_vectors_size, valid_vectors_ngram, valid_vectors_ngram_pos, valid_vectors_liwc, 
                    valid_vectors_tutoring_moves, valid_vectors_pvb, valid_vectors_nvb, valid_vectors_precision)]

            # Remove NaN and inf from the features

            train_kdf_features = np.array(train_kdf_features)
            valid_kdf_features = np.array(valid_kdf_features)

            train_kdf_features[np.isnan(train_kdf_features)] = 0.0
            valid_kdf_features[np.isnan(valid_kdf_features)] = 0.0

            train_kdf_features[np.isfinite(train_kdf_features)==False] = 1.0
            valid_kdf_features[np.isfinite(valid_kdf_features)==False] = 1.0

        if args.model in ["pte", "pte+kdf"]:
            # Embeddings
            vectors_embeddings = list(train_features["Embeddings"])
            valid_vectors_embeddings = list(valid_features["Embeddings"])                

        # Concatenate with the embeddings

        if args.model == "pte+kdf":
            train_features = np.array([np.concatenate((v_embed, v_feat)) for v_embed, v_feat in \
                zip(vectors_embeddings, train_kdf_features)])
            valid_features = np.array([np.concatenate((v_embed, v_feat)) for v_embed, v_feat in \
                zip(valid_vectors_embeddings, valid_kdf_features)])
        
        elif args.model == "pte":
            train_features = np.array([np.concatenate(([v_size], v_emb)) for v_size, v_emb in \
                zip(vectors_size, vectors_embeddings)])
            valid_features = np.array([np.concatenate(([v_size], v_emb)) for v_size, v_emb in \
                zip(valid_vectors_size, valid_vectors_embeddings)])

        else :
            train_features = train_kdf_features
            valid_features = valid_kdf_features

        # Balancing of classes
        class_weights = compute_class_weight("balanced", classes = np.unique(train_labels), y=train_labels)
        class_weights = [np.sqrt(class_weights[-1]), class_weights[0], np.sqrt(class_weights[2]), np.sqrt(class_weights[1])]

        if args.run_mode == "hyper_opt":

            def define_model(trial):

                p = trial.suggest_float("dropout", 0.15, 0.25)
                layer_1_hidden_size = trial.suggest_int("layer_1_hidden_size", 32, 64)
                layer_2_hidden_size = trial.suggest_int("layer_2_hidden_size", 16, 32)
                layers = []

                in_features = len(train_features[0])

                layers.append(nn.Linear(in_features, layer_1_hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p))

                layers.append(nn.Linear(layer_1_hidden_size, layer_2_hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p))

                layers.append(nn.Linear(layer_2_hidden_size, CLASSES))
                layers.append(nn.LogSoftmax(dim=1))
                return nn.Sequential(*layers)

                # Define an objective function to be minimized.

            def objective(trial, train_features, train_labels, valid_features, valid_labels, mapping_indirectness, class_weights):

                train_labels = [inverse_mapping_indirectness[x] for x in train_labels]
                train_dataset = IndirectnessDataset(train_features, train_labels)
                train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = False)

                valid_labels = [inverse_mapping_indirectness[x] for x in valid_labels]
                valid_dataset = IndirectnessDataset(valid_features, valid_labels)
                valid_dataloader = DataLoader(valid_dataset, batch_size = BATCHSIZE, shuffle = False)
                
                model = define_model(trial)
                lr = trial.suggest_float("lr", 1e-4, 1e-2)
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
                    labels = ["IDA", "IDS", "IDQ"],
                    average="weighted")
                
                print(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))

                return f1_score
                
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial:objective(trial, train_features, train_labels, valid_features,\
                 valid_labels, mapping_indirectness, class_weights), n_trials=100)
            print(study.best_trial)
        
        elif args.run_mode == "eval":

            # Training after the hyper-optimization process

            train_labels = [inverse_mapping_indirectness[x] for x in train_labels]
            train_dataset = IndirectnessDataset(train_features, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = False)

            valid_labels = [inverse_mapping_indirectness[x] for x in valid_labels]
            valid_dataset = IndirectnessDataset(valid_features, valid_labels)
            valid_dataloader = DataLoader(valid_dataset, batch_size = BATCHSIZE, shuffle = False)

            layers = [nn.Linear(len(train_features[0]), 34), nn.ReLU(), nn.Dropout(0.18), \
            nn.Linear(34, 25), nn.ReLU(), nn.Dropout(0.18), \
            nn.Linear(25, CLASSES), nn.LogSoftmax(dim=1)]

            model = nn.Sequential(*layers)
            
            lr = 0.0006
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

            all_classes_f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels, average="weighted"))
            f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels,
                labels = ["IDA", "IDS", "IDQ"],
                average="weighted"))
            classification_reports.append(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))
            confusion_matrices.append(confusion_matrix(valid_labels, pred_labels))
        
    if args.run_mode == "eval":    
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
