# General imports
import matplotlib
import matplotlib.pyplot as plt
import json

# Scikit-learn imports
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

def globalConfusionMatrix(label, predict):
    display_labels = ["find-train","find-flight","find-restaurant","purchase","find-around-me","provide-showtimes","find-hotel","irrelevant"]
    cm = confusion_matrix(label, predict, labels=display_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    disp.plot(xticks_rotation='vertical')