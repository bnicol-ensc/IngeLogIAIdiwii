import spacy
import en_core_web_sm
import json
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
from spacy.util import minibatch, compounding

def import_data(verbose = True):
    json_file = open('../data/processed/training_set.json', encoding='utf8')
    data = json.load(json_file)
    if verbose:
        print("Data has been imported")
    
    y_training_raw = np.array([datapoint["intent"] for datapoint in data])
    x_training = np.array([datapoint["sentence"] for datapoint in data])
    encoder = OneHotEncoder(sparse=False)
    y_training = encoder.fit_transform(y_training_raw.reshape(-1,1))
    
    if verbose:
        print("Data has been converted from : \n", encoder.inverse_transform(y_training[:10,:]))
        print("\nto one-hot encoded format : \n", y_training[:10,:])
        
        print("example data point : ",y_training[2], " : ",x_training[2])
        
    assert len(y_training) == len(x_training)
    return x_training, y_training, encoder

    
def build_model(encoder, verbose = True):
    nlp = en_core_web_sm.load()
    # Adding the built-in textcat component to the pipeline.
    textcat=nlp.create_pipe( "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
    nlp.add_pipe(textcat, last=True)
    nlp.pipe_names
    
    # Add categories to the tagger
    for cat in encoder.categories_[0]:
        if verbose == True:
            print(cat)
        textcat.add_label(cat)
    return nlp
        
def treat_data(x_training, y_training, encoder):
    list_cats_dic = [{'cats' : {encoder.categories_[0][i] : cat for i, cat in enumerate(cats)}} for cats in y_training]
    str_x_training = [str(x) for x in x_training] # spacy cannot handle numpy.str_ type which was used to create the list

    train_data = list(zip(str_x_training,list_cats_dic))
    with open('../data/processed_spacy/training_data.json', 'w') as f:
        json.dump(train_data, f)
    return train_data
        
def train(train_data, nlp, n_iter=10, verbose = True):
    # Disabling other components
    n_iter=10
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        # Performing training
        for i in range(n_iter):
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            if verbose:
                print(losses)
    nlp.to_disk('../models/model1')
    if verbose:
        print("model trained and saved")

if __name__ == "__main__":
    x_training, y_training, encoder = import_data()
    nlp = build_model(encoder)
    train_data = treat_data(x_training, y_training, encoder)
    train(train_data, nlp)