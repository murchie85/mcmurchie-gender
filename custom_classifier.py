"""
Classify Gender from a name
USING CUSTOM FEATURE ANALYSIS
# FLOW: Features function, apply funciton, vectorise, fit, transform, classify, fit, predict
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
#						CUSTOM FEATURE FUNCTION
#====================================================================================
# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

#====================================================================================
#						CUSTOM FEATURE FUNCTION & EXTRACTION
#====================================================================================
features = np.vectorize(features)
print('Printing dict for ' + str(["Anna", "Hannah", "Peter","John","Vladmir","Mohammed"]))
print(features(["Anna", "Hannah", "Peter","John","Vladmir","Mohammed"]))

df_X = features(df_names['name'])
df_y = df_names['sex']

#====================================================================================
#						CUSTOM FEATURE FUNCTION & EXTRACTION
#====================================================================================

from sklearn.feature_extraction import DictVectorizer
 
corpus = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(corpus)
print('Printing Mike and Julia applied to dict and DictVectorised then transformed')
print(transformed)

dv.get_feature_names()

#====================================================================================
#						SETTING X/Y TRAIN/TEST
#====================================================================================
from sklearn.model_selection import train_test_split
# Train Test Split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dfX_train



#====================================================================================
#						FITTING
#====================================================================================
dv = DictVectorizer()
dv.fit_transform(dfX_train)

vectorizer = dv.fit(dfX_train)


#====================================================================================
#						TRAINING DECISION TREE
#====================================================================================

from sklearn.tree import DecisionTreeClassifier
 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)



#====================================================================================
#						TRANSFORM AND PREDICT
#====================================================================================



def predict(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        print(str(a) + " is Female")
        return("Female")
    else:
        print(str(a) + " is Male")
        return("Male")

random_name_list = ["Alex","Alice","Chioma","Vitalic","Clairese","Chan"]

for n in random_name_list:
	predict(n)



#====================================================================================
#						ACCURACY
#====================================================================================
# Accuracy on training set
print("Training accuracy is ", dclf.score(dv.transform(dfX_train), dfy_train))
# Accuracy on test set
print("Test accuracy is ", dclf.score(dv.transform(dfX_test), dfy_test))
#---------------------------------------
#		SAVE MODEL
#---------------------------------------

import joblib
decisiontreModel = open("models/decisiontree.pkl","wb")
joblib.dump(dclf,decisiontreModel)
decisiontreModel.close

"""
#Alternative to Model Saving
import pickle
dctreeModel = open("models/namesdetectormodel.pkl","wb")
pickle.dump(dclf,dctreeModel)
dctreeModel.close()
"""

vectorOut = open("models/DCvectorizer.pkl","wb")
joblib.dump(vectorizer, vectorOut)
