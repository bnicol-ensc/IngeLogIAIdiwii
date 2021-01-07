# General imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

from scipy import interp
from itertools import cycle

# Scikit-learn imports
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.preprocessing import OneHotEncoder

# Getting the total number of items in a given data
def totalNumberOfItems(data):
    return len(data)

# Computing the items for each class
def itemsInClasses(data):
    classes = {"find-train":0,
           "find-flight":0,
           "find-restaurant":0,
           "purchase":0,
           "find-around-me":0,
           "provide-showtimes":0,
           "find-hotel":0,
           "irrelevant":0}

    for i in range(len(data)):
        intent = data[i]['intent']
        classes[intent] = classes.get(intent, 0) + 1

    return classes

# Pie plot for classes distribution
def classesDistribution(data, classes):
    classesProp = [x/len(data) for x in classes.values()]

    fig = plt.figure()
    fig.patch.set_facecolor('black')

    plt.rcParams['text.color'] = 'white'
    plt.pie(classesProp, labels=classes.keys(), autopct='%1.1f%%', shadow=True, radius=2)
    plt.show()

# Get all sentences
def getSentence(data):
    return [x["sentence"] for x in data]

# Get all intents
def getIntent(data):
    return [x["intent"] for x in data]

# Compute precision, recall and f1
def computeMetrics(label, predict):
    # precision tp / (tp + fp)
    precision = precision_score(label, predict, average="weighted")
    
    # recall: tp / (tp + fn)
    recall = recall_score(label, predict, average="weighted")
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(label, predict, average="weighted")
    
    return precision, recall, f1

def plotGlobalConfusionMatrix(label, predict, irrelevant=True):
    display_labels = ["find-train","find-flight","find-restaurant","purchase","find-around-me","provide-showtimes","find-hotel"]
    if irrelevant:
        display_labels.append("irrelevant")

    cm = confusion_matrix(label, predict, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(xticks_rotation='vertical')

# Compute ROC curve and ROC area for each class
def plotROC(results, testIntent, resultsMax):
    classes_name = list(results[0].keys())
    y_score = np.array([list(x.values()) for x in results])
    y_true = np.array(testIntent)
    y_pred = np.array(resultsMax)

    encoder = OneHotEncoder(sparse=False)
    y_enc_true = np.array(encoder.fit_transform(y_true.reshape(-1,1)))
    y_enc_pred = np.array(encoder.transform(y_pred.reshape(-1,1)))

    fpr = dict()
    tpr = dict()
    rocauc = dict()
    for i in range(8):
        fpr[i], tpr[i], _  = roc_curve(y_enc_true[:, i], y_score[:, i])
        rocauc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"],  _ = roc_curve(y_enc_true.ravel(), y_score.ravel())
    roc_auc = dict()
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
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
    plt.rcParams['text.color'] = 'black'
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    lw = 1
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(classes_name[i], rocauc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC curves')
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.show()

def threshold(results, testIntent):
    y_score = np.array([list(x.values()) for x in results])
    y_pred = np.array([list(x.key()) for x in results])
    y_true = np.array(testIntent)
    
    true = []
    false = []
    for i,score in enumerate(y_score):
        if y_true[i] == y_pred[i]:
            true.append(np.argmax(score))
        else:
            false.append(np.argmax(score))
            
    print("mean activation score when classification correct : ",np.mean(np.array(true)))
    print("mean activation score when classification wrong : ",np.mean(np.array(false)))