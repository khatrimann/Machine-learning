#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:14:22 2018

@author: mann
"""

# Naive Bayes

# Importing hte libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

# Copied function for finding top words in respective files
def show_top10(classifier, vectorizer, categories):
     feature_names = np.asarray(vectorizer.get_feature_names())
     for i, category in enumerate(categories):
         top10 = np.argsort(classifier.coef_[i])[-10:]
         print("%s: %s" % (category, " ".join(feature_names[top10])))
         


# Importing dataset and splitting into Test and Train sets
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
categories = list(newsgroups_train.target_names)

# Converting text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

# Fitting test data to Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.015)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

# Using pipelining on the same
from sklearn.pipeline import make_pipeline
model = make_pipeline(vectorizer, clf)
model.fit(newsgroups_train.data, newsgroups_train.target)
pipe_pred = model.predict(newsgroups_test.data)
metrics.f1_score(newsgroups_test.target, pipe_pred, average='macro')

# Predicting
new_pred = model.predict(["The multi-launch contract with Spire - a company providing weather, maritime, and aviation data to public and private customers - will cover a significant number of CubeSats to be launched on Vega as part of the Small Spacecraft Mission Service Proof Of Concept (POC) flight in 2019, as well as options on subsequent Vega flights. With more than 80 satellites placed in orbit during the past four years, Spire has quickly become an important leader in the New Space community.Built in-house by Spire using its LEMUR2 CubeSat platform, the nanosatellites will weigh approximately 5 kg. at launch and are designed to have a nominal service life of two to three years once positioned in a Sun-synchronous orbit at 500 km. Each satellite carries multiple sensors, making them capable of performing data collection for all of Spire's data products."])

#Confusion Matrix
import seaborn as sns
sns.set()
mat = metrics.confusion_matrix(newsgroups_test.target, pipe_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=newsgroups_train.target_names, yticklabels=newsgroups_train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')