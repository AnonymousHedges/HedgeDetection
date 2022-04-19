# Perform the SHAP features 

import shap
import lightgbm as lgb
import numpy as np 
import pandas as pd
import optuna
import sklearn
from sklearn.model_selection import StratifiedKFold
from vector_features_sentence import get_liwc, get_ngram, get_ngram_pos_tag, category_names
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from ast import literal_eval
from classifier_rule_based_no_syntax import classify_sentence, precision_rules
from sklearn.metrics import confusion_matrix
import pickle as pkl

if __name__ == "__main__":

    # Load the dataset

    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] if x.split()[0] != "IDE" else "IDQ" for x in list(dataset["Label"])]

    mapping_indirectness = {
        (1, 0, 0, 0) : "Nothing",
        (0, 1, 0, 0) : "IDA",
        (0, 0, 1, 0) : "IDS", 
        (0, 0, 0, 1) : "IDQ",
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
    dataset["Nonverbal Behaviors"] = dataset["Nonverbal Behaviors"].apply(literal_eval)

    # Replace the columns with the mapped values

    new_values = [mapping_tutoring_moves[x] if x in mapping_tutoring_moves else (0, 0, 0, 0, 0, 0) for x in dataset["Tutoring Moves"]] 
    dataset.drop(columns=["Tutoring Moves"], axis = 1, inplace = True)
    dataset["Tutoring Moves"] = new_values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 1)
    X = dataset.drop(columns=["Label"])
    Y = dataset["Label"]
    skf.get_n_splits(X, Y)

    train_indexes, valid_indexes = next(skf.split(X,Y))

    train_features, valid_features = X.iloc[train_indexes], X.iloc[valid_indexes]
    train_labels, valid_labels = list(Y.iloc[train_indexes]), list(Y.iloc[valid_indexes])

    train_texts = [literal_eval(x)[1] for x in train_features["Text"]]
    valid_texts = [literal_eval(x)[1] for x in valid_features["Text"]]

    # Ngrams

    train_dict_ngram, vectors_ngram = get_ngram(train_texts, train=True)
    valid_vectors_ngram = get_ngram(valid_texts, train=False, runtime_dict=train_dict_ngram)

    # Pos-tag

    train_dict_pos, vectors_ngram_pos = get_ngram_pos_tag(train_texts, train=True)
    valid_vectors_ngram_pos = get_ngram_pos_tag(valid_texts, train=False, runtime_dict=train_dict_pos)


    # Size clauses

    vectors_size = [len(word_tokenize(x)) for x in train_texts]
    valid_vectors_size = [len(word_tokenize(x)) for x in valid_texts]

    # LIWC 

    vectors_liwc = [get_liwc(x) for x in train_texts]
    valid_vectors_liwc = [get_liwc(x) for x in valid_texts]

    # Tutoring moves 

    vectors_tutoring_moves = list(train_features["Tutoring Moves"])
    valid_vectors_tutoring_moves = list(valid_features["Tutoring Moves"])

    # Non-verbal behaviors 

    vectors_nvb = list(train_features["Nonverbal Behaviors"])
    valid_vectors_nvb = list(valid_features["Nonverbal Behaviors"])

    # Concatenate features

    dict_precision = precision_rules(train_texts, train_labels)
    dict_precision["Nothing"] = {"0":1.0}
    vectors_precision = [classify_sentence(sent) for sent in train_texts]
    vectors_precision = [np.array(inverse_mapping_indirectness[x[0]])*dict_precision[x[0]][x[1]] for x in vectors_precision]
    valid_vectors_precision = [classify_sentence(sent) for sent in valid_texts]
    valid_vectors_precision = [np.array(inverse_mapping_indirectness[x[0]])*dict_precision[x[0]][x[1]] for x in valid_vectors_precision]

    train_features = [np.concatenate((
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
        # Nonverbal Behaviors
        np.array(v_nvb),
        # Precision
        np.array(v_precision)
    )) for v_size, v_ngram, v_ngram_pos, v_liwc, v_tutoring_moves, v_nvb, v_precision in 
    zip(vectors_size, vectors_ngram, vectors_ngram_pos, vectors_liwc, vectors_tutoring_moves, vectors_nvb, vectors_precision)]

    valid_features = [np.concatenate((
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
        # Nonverbal Behaviors
        np.array(v_nvb),
        # Precision
        np.array(v_precision)
    )) for v_size, v_ngram, v_ngram_pos, v_liwc, v_tutoring_moves, v_nvb, v_precision in 
    zip(valid_vectors_size, valid_vectors_ngram, valid_vectors_ngram_pos, valid_vectors_liwc, 
            valid_vectors_tutoring_moves, valid_vectors_nvb, valid_vectors_precision)]

    # Remove NaN and inf from the features

    train_features = np.array(train_features)
    valid_features = np.array(valid_features)

    train_features[np.isnan(train_features)] = 0.0
    valid_features[np.isnan(valid_features)] = 0.0

    train_features[np.isfinite(train_features)==False] = 1.0
    valid_features[np.isfinite(valid_features)==False] = 1.0

    # Balancing of classes

    sample_weights = compute_sample_weight("balanced", train_labels)

    sample_weights = np.sqrt(np.sqrt(sample_weights))   

    #Precision + ngram + pos-tag + liwc + tutoring moves + nvb = all features
    parameters = {"n_estimators" : 2000, 
            "num_leaves" : 75,
            "learning_rate" : 0.0024,
            "reg_alpha" : 0.424,
            "reg_lambda" : 0.500,
            "objective": "multi:softmax",
            "n_jobs" : 6,
            "verbose" : -1
        } 

    lgb_model = lgb.LGBMClassifier(**parameters)
    lgb_model.fit(train_features, train_labels, eval_set=[(valid_features, valid_labels)], early_stopping_rounds=20,
            sample_weight = sample_weights, verbose=-1)

    pred_labels = lgb_model.predict(valid_features, verbose=-1)

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(np.concatenate((train_features, valid_features)))

    feature_names = ["Clause length"] + [str(x) for x in list(train_dict_ngram[1].keys())] + [str(x) for x in list(train_dict_ngram[2].keys())] + \
    [str(x) for x in list(train_dict_pos[1].keys())] + [str(x) for x in list(train_dict_pos[2].keys())] + [str(x) for x in list(train_dict_pos[3].keys())] + \
    [str(x) for x in category_names] + list(string_mapping_tutoring_moves.keys())  + columns_nonverbal_behaviors + \
    ["Label_Nothing", "Label_IDA", "Label_IDS", "Label_IDQ"]

    print("All")
    shap.summary_plot(shap_values, np.concatenate((train_features, valid_features)), feature_names=feature_names)
    breakpoint()
    print("IDA")
    shap.summary_plot(shap_values[0], np.concatenate((train_features, valid_features)), feature_names=feature_names)
    breakpoint()
    print("IDQ")
    shap.summary_plot(shap_values[1], np.concatenate((train_features, valid_features)), feature_names=feature_names)
    breakpoint()
    print("IDS")
    shap.summary_plot(shap_values[2], np.concatenate((train_features, valid_features)), feature_names=feature_names)
    breakpoint()
    print("Nothing")
    shap.summary_plot(shap_values[3], np.concatenate((train_features, valid_features)), feature_names=feature_names)