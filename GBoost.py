
import lightgbm as lgb
import numpy as np 
import pandas as pd
import optuna
import sklearn
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from vector_features_sentence import get_liwc, get_ngram, get_ngram_pos_tag
from sklearn.utils.class_weight import compute_sample_weight
from ast import literal_eval
from classifier_rule_based_no_syntax import classify_sentence, precision_rules
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action="store", help = "Choose the model representation, between Pretrained-Embeddings \
        (pte), Knowledge-Driven Features (kdf), or the combination of both (pte+kdf)")
    parser.add_argument('--run_mode', action="store", help = "Choose whether to evaluate the model \
        ('eval') or to search for hyper-parameters ('hyper_opt')")
    args = parser.parse_args()
    
    # Load the dataset

    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] for x in list(dataset["Label"])]

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

    # Cut the dataset in sub-parts

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
        
        # Extract part of the dataset

        train_features, valid_features = X.iloc[train_indexes], X.iloc[valid_indexes]
        train_labels, valid_labels = list(Y.iloc[train_indexes]), list(Y.iloc[valid_indexes])

        train_texts = [literal_eval(x)[1] for x in list(train_features["Text"])]
        valid_texts = [literal_eval(x)[1] for x in list(valid_features["Text"])]

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

            # Construct features

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

        if args.model in ["pte+kdf"]:

            # Try with PCA
            """
            sparse_train_features = csr_matrix(np.array(train_features))
            sparse_valid_features = csr_matrix(np.array(valid_features))
            svd = TruncatedSVD(n_components=100, algorithm = "arpack")
            train_features = svd.fit_transform(sparse_train_features)
            valid_features = svd.transform(sparse_valid_features)
            """
            # Concatenate with the embeddings

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

        # Put the classes balancing

        sample_weights = compute_sample_weight("balanced", train_labels)
        sample_weights = np.sqrt(np.sqrt(sample_weights))

        if args.run_mode == "hyper_opt":
            def objective(trial, train_features, train_labels, valid_features, valid_labels, sample_weights, mapping_indirectness, inverse_mapping_indirectness):

                parameters = {"n_estimators" : 3000, 
                        "num_leaves" : trial.suggest_int("num_leaves", 64, 256),
                        "learning_rate" : trial.suggest_float("learning_rate", 5e-3, 1e-1),
                        "reg_alpha" : trial.suggest_float("reg_alpha", 0.0, 0.5),
                        "reg_lambda" : trial.suggest_float("reg_lambda", 0.0, 0.5),
                        "objective": "multi:softmax",
                        "n_jobs" : 6,
                        "verbose" : -1
                    }
                    
                lgb_model = lgb.LGBMClassifier(**parameters)
                lgb_model.fit(train_features, train_labels, eval_set=[(valid_features, valid_labels)], early_stopping_rounds=20,
                    sample_weight = sample_weights, verbose=-1)

                pred_labels = lgb_model.predict(valid_features, verbose=-1)

                f1_score = sklearn.metrics.f1_score(valid_labels, pred_labels, labels = ["IDA", "IDQ", "IDS"], average="weighted")
                all_classes_f1_score = sklearn.metrics.f1_score(valid_labels, pred_labels, labels = ["IDA", "IDQ", "IDS"], average="weighted")
                print(classification_report(valid_labels, pred_labels, labels=["IDQ", "IDS", "IDA"]))
                print("Weighted All classes F1 score = "+str(all_classes_f1_score))
                return f1_score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, train_features, train_labels, valid_features, valid_labels, sample_weights, mapping_indirectness, inverse_mapping_indirectness), n_trials=100)
            print(study.best_trial)

        elif args.run_mode == "eval":

            parameters = {"n_estimators" : 3000, 
                    "num_leaves" : 106,
                    "learning_rate" : 0.041,
                    "reg_alpha" : 0.143,
                    "reg_lambda" : 0.296,
                    "objective": "multi:softmax",
                    "n_jobs" : 6,
                    "verbose" : -1
                } 

            lgb_model = lgb.LGBMClassifier(**parameters)
            lgb_model.fit(train_features, train_labels, eval_set=[(valid_features, valid_labels)], early_stopping_rounds=20,
                    sample_weight = sample_weights, verbose=-1)

            pred_labels = lgb_model.predict(valid_features, verbose=-1)

            all_classes_f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels, average="weighted"))
            f1_scores.append(sklearn.metrics.f1_score(valid_labels, pred_labels, labels = ["IDA", "IDQ", "IDS"], average="weighted"))
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
