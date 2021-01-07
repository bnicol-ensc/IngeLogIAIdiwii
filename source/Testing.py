from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, f1_score, fbeta_score
from sklearn.preprocessing import OneHotEncoder
from scipy import interp
from itertools import cycle

import spacy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os 
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nlp = spacy.load(f"{d}/models/model1")
textcat = nlp.get_pipe('textcat')

def predict(sentence = "What does life even mean ?"):
    docs = nlp.tokenizer(sentence)
    # Use textcat to get the scores for each doc
    scores, _ = textcat.predict([docs])
    label = scores.argmax()
    return scores, textcat.labels[label]

def compute_predictions():
    # Reading training data
    with open(f'{d}/data/processed/testing_set.json',encoding="utf8") as f:
        testingData = json.load(f)
    # Creating a dataframe with the training data
    y_true = np.array([expected['intent'] for expected in testingData])
    y_pred = np.array([predict(expected['sentence'])[1] for expected in testingData])
    y_score = np.array([predict(expected['sentence'])[0][0] for expected in testingData])
    return y_true,y_pred,y_score

def display_sentence_example():
    # Reading training data
    with open(f'{d}/data/processed/testing_set.json',encoding="utf8") as f:
        testingData = json.load(f)
    # Creating a dataframe with the training data
    y_true, y_pred,y_score = compute_predictions()
    print(y_true[1], y_pred[1], y_score[1])

def display_recall():
    y_true, y_pred,_ = compute_predictions()
    print("recall :")
    print("micro : \t", recall_score(y_true, y_pred, average = "micro"))
    print("weighted : \t", recall_score(y_true, y_pred, average = "weighted"))
    print("macro : \t",recall_score(y_true, y_pred, average = "macro"))

def display_f1_score():
    y_true, y_pred,_ = compute_predictions()
    print("f1 score : ")
    print("micro : \t", f1_score(y_true, y_pred, average = "micro"))
    print("weighted : \t", f1_score(y_true, y_pred, average = "weighted"))
    print("macro : \t",f1_score(y_true, y_pred, average = "macro"))

def display_accuracy_score():
    y_true, y_pred,_ = compute_predictions()
    print("accuracy score :\t", accuracy_score(y_true, y_pred))

def display_fbeta_score():
    y_true, y_pred,_ = compute_predictions()
    fbeta_score
    print("fbeta score : ")
    print("micro : \t", fbeta_score(y_true, y_pred, beta=0.5, average = "micro"))
    print("weighted : \t", fbeta_score(y_true, y_pred, beta=0.5, average = "weighted"))
    print("macro : \t",fbeta_score(y_true, y_pred, beta=0.5, average = "macro"))

def compute_roc_curve():
    # Compute ROC curve and ROC area for each class    
    encoder = OneHotEncoder(sparse=False)
    y_true,_,y_score = compute_predictions()
    y_enc_true = np.array(encoder.fit_transform(y_true.reshape(-1,1)))
    # y_enc_pred = np.array(encoder.transform(y_pred.reshape(-1,1)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_enc_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_enc_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr,tpr,roc_auc


def plot_roc():
    fpr,tpr,roc_auc = compute_roc_curve()
    n_classes = 8
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()