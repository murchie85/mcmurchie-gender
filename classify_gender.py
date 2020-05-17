"""
Classify Gender from a name
"""

#====================================================================================
#						IMPORTS & DATA LOADING
#====================================================================================


# EDA packages Exploratory data Analysis

import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

# Load our data
df = pd.read_csv('data/names.csv')

#====================================================================================
#						INSPECTION
#====================================================================================


df.head()
df.size
#check column types
df.columns
df.dtypes


# Checking for Missing Values
df.isnull().isnull().sum()


# Number of Female Names
df[df.sex == 'F'].size
# Number of Male Names
df[df.sex == 'M'].size


#====================================================================================
#						CLEANING
#====================================================================================

df_names = df
# Replacing All F and M with 0 and 1 respectively
df_names.sex.replace({'F':0,'M':1},inplace=True)

#unique values
df_names.sex.unique()

df_names.dtypes


#====================================================================================
#						FEATURE EXTRACTION
#====================================================================================

Xfeatures =df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

print('Get all names')
cv.get_feature_names()

#====================================================================================
#						SETTING X/Y TRAIN/TEST
#====================================================================================
from sklearn.model_selection import train_test_split

# Features 
X
# Labels
y = df_names.sex


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#====================================================================================
#						# Naive Bayes Classifier
#====================================================================================

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

#---------------------------------------
#		PREDICTIONS
#---------------------------------------

vect = cv.transform(["Mary"]).toarray()

# Uncomment to see vector structure 
print('Mary as vector is..')
print(len(vect))
print('')
# Female is 0, Male is 1
print('Predicting if Mary is a boy or girl')
print('0=girl, 1=boy')
print(clf.predict(vect))


