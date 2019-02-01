# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:50:49 2019

@author: HeavyD
"""

import nltk
import sklearn
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words("english")
import re
import numpy as np
import pandas as pd
import os
def dataloader(datadir):
    data={}
    for split in ["train","test"]:
        data[split]=[]
        for sentiment in ["neg","pos"]:
            if sentiment=="pos":
                score=1
            else:
                score=0
            path=os.path.join(datadir,split,sentiment)
            file_names=os.listdir(path)
            for fname in file_names:
                with open(os.path.join(path,fname),"r",encoding="utf8") as f:
                    review=f.read();
                    data[split].append([review,score])
    np.random.shuffle(data["train"]);
    data["train"]=pd.DataFrame(data["train"],columns=["text","sentiment"])
    data["test"]=pd.DataFrame(data["test"],columns=["text","sentiment"])
    return data["train"],data["test"]
trainset,testset=dataloader(path to data set folder)
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Transform each text into a vector of word counts
vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 2))

training_features = vectorizer.fit_transform(trainset["text"])    
test_features = vectorizer.transform(testset["text"])

# Training
model = LinearSVC()
model.fit(training_features, trainset["sentiment"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

