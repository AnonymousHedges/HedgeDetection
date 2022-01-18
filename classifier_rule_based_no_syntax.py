import subprocess
import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from ast import literal_eval
import sklearn
from sklearn.metrics import confusion_matrix

pattern_shield = ["(?!what).*(i|we) ?(don't|didn't|did)? ?(not)? (guess|guessed|thought|think|believe|believed|suppose|supposed) ?(whether|if|is|that|it|this)?.*",
    ".*(i|i'm|we) ?(was|am|wasn't)? ?(not)? (sure|certain).*",
    ".*(i feel like you).*",
    ".*(you (might|may) (believe|think)).*"
    ".*(according to|presumably).*",
    ".*(i|you|we) have to (check|look|verify).*",
    ".*(if i'm not wrong|if i'm right|if that's true).*",
    ".*(unless i).*"]

pattern_apologies = [".*(i'm|i|we're) (am|are)? ?(apologize|sorry).*",
"(?!.*(be|been|was) like excuse me)((excuse me|sorry)[\w ,']+|[\w ,']+(excuse me|sorry))"]
# Things like "I'm sorry" or "Sorry", where the apology is the sole purpose of the utterance should not be included
list_exceptions_ida = ["i'm sorry", "oh sorry", "right sorry"]

pattern_extenders = [".* (or|and) (that|something|stuff|so forth)",
".*, (weren't you\?|were you\?|aren't i)",
".*(all that|whatever)"]

pattern_qualifiers = [".*(just|a little|maybe|actually|sort of|kind of|pretty much|somewhat|exactly|almost|little bit|quite|\
regular|regularly|actually|almost|as it were|basically|probably|can be view as|crypto-|especially|essentially|exceptionally|for the most part|in a manner of speaking|\
in a real sense|in a sense|in a way|largely|literally|loosely speaking|kinda|more or less|mostly|often|on the tall side|par excellence|particularly|pretty much|principally|\
pseudo-|quintessentially|relatively|roughly|so to say|strictly speaking|technically|typically|virtually|approximately|something between|essentially|only).*",
".*(i|i'm|you|it's) (am|are) (apparently|surely)[ ,]?.*",
".*(it) (looks|seems|appears)[ ,]?.*"]

# Transform the transcripts into a set of sentence / labels (there can be multiple labels for one sentence)

def print_errors(list_sentences, list_pred, list_true, categories):
    for sentence, pred, true in zip(list_sentences, list_pred, list_true):
        #if re.match(".*you know.*", sentence, re.IGNORECASE):
        #  or pred in categories
            if pred != true and (pred in categories): 
                print("Sentence: " + sentence)
                print("Pred label :" + pred)
                print("True label :" + true)
                print()

def preprocess_sentence(sent):
    # Remove noise like "sfx", laughter
    return " ".join([x for x in sent.split() if x not in ["sfx", "laughter", "okay", "inaudible", "inhale", "inhales", "exhale", "exhales", "um"]]).strip()

def read_transcripts(transcripts):

    columns = set([y for x in transcripts for y in x.columns])
    id_columns = [x for x in columns if re.match("ID_.*",x)]
    
    sentences, labels = [], []
    for t in transcripts:
        for _,r in t.iterrows():
            if type(r["P1"]) != float:
                sentences.append("".join([x for x in r["P1"] if x not in ["[","]","-","(",")", "{", "}"]]))
            elif type(r["P2"]) != float:
                sentences.append("".join([x for x in r["P2"] if x not in ["[","]","-","(",")", "{", "}"]]))
            else:
                continue
            absense_label = True
            for c in id_columns:
                try:
                    if type(r[c]) != float:
                        labels.append(r[c].strip())
                        absense_label = False
                except:
                    pass
            if absense_label:
                labels.append("Nothing")
    assert len(sentences) == len(labels)
    return sentences, labels    

def classify_sentence(sent):
    labels = []
    if len(sent.split()) > 1 :

        for i, p in enumerate(pattern_shield):
            if re.match(p, sent, re.IGNORECASE):
                labels.append(("IDS", i))
                break

        for i, p in enumerate(pattern_apologies):
            if re.match(p, sent, re.IGNORECASE) and sent.lower() not in list_exceptions_ida:
                labels.append(("IDA", i))
                break

        for i, p in enumerate(pattern_extenders):
            if re.match(p, sent, re.IGNORECASE):
                # We will consider the Extenders as Propositional Hedges
                labels.append(("IDQ", len(pattern_qualifiers) + i))
                break

        for i, p in enumerate(pattern_qualifiers):
            if re.match(p, sent, re.IGNORECASE):
                labels.append(("IDQ", i))
                break

    if len(labels) > 0:
        return labels[0]
    else:
        return ("Nothing", "0")

def precision_rules(sentences, labels):

    # Precision (TP/FP+TP) of each rule
    dict_precision = {"IDA":{k:0.0 for k in range(len(pattern_apologies))}, "IDQ":{k:0.0 for k in range(len(pattern_qualifiers) + len(pattern_extenders))},\
        "IDS":{k:0.0 for k in range(len(pattern_shield))}}
    count = Counter(labels)
    total_IDA = count["IDA"]
    total_IDS = count["IDS"]
    total_IDQ = count["IDQ"] + count["IDE"]

    for sent, label in zip(sentences, labels):
        
        if len(sent.split()) > 1 :
            for i, p in enumerate(pattern_shield):
                if re.match(p, sent, re.IGNORECASE):
                    dict_precision["IDS"][i] += 1
            
            for i, p in enumerate(pattern_apologies):
                if re.match(p, sent, re.IGNORECASE) and sent.lower() not in list_exceptions_ida:
                    dict_precision["IDA"][i] += 1

            for i, p in enumerate(pattern_extenders):
                if re.match(p, sent, re.IGNORECASE):
                    dict_precision["IDQ"][len(pattern_qualifiers) + i] += 1
            
            for i, p in enumerate(pattern_qualifiers):
                if re.match(p, sent, re.IGNORECASE):
                    dict_precision["IDQ"][i] += 1

    try:
        dict_precision["IDA"] = {k:v/total_IDA for k,v in dict_precision["IDA"].items()}
    except:
        dict_precision["IDA"] = {k:0.0 for k,v in dict_precision["IDA"].items()}
    dict_precision["IDQ"] = {k:v/total_IDQ for k,v in dict_precision["IDQ"].items()}
    dict_precision["IDS"] = {k:v/total_IDS for k,v in dict_precision["IDS"].items()}

    return dict_precision

if __name__=="__main__":
    
    dataset = pd.read_csv("indirectness_dataset.csv")
    dataset = dataset[(dataset["Period"] == "T1") | (dataset["Period"] == "T2")]

    # Remove the multiple labels and replace IDE with IDQ
    dataset["Label"] = [x.split()[0] if x.split()[0] != "IDE" else "IDQ" for x in list(dataset["Label"])]

    pred_labels = [classify_sentence(preprocess_sentence(literal_eval(sent)[1]))[0] for sent in list(dataset["Text"])]  
    print(sklearn.metrics.f1_score(list(dataset["Label"]), pred_labels, labels = ["IDA", "IDS", "IDQ"], average="weighted"))  
    print(sklearn.metrics.f1_score(list(dataset["Label"]), pred_labels, average="weighted"))  
    print(classification_report(list(dataset["Label"]), pred_labels))
    print(classification_report(list(dataset["Label"]), pred_labels, labels=["IDQ", "IDS", "IDA"]))
    print(confusion_matrix(list(dataset["Label"]), pred_labels))
