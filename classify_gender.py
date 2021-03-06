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

# Required for importing
vectorizer = cv.fit(Xfeatures)

#print('Get all names')
#cv.get_feature_names()

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

"""
#---------------------------------------
#		PREDICTIONS FOR INFO
#---------------------------------------
# First cv = countvectorizer, then we do X = cv.fit_transform(Xfeatures)... 
# (even tho it equals x, it is still fitted)now this.
vect = cv.transform(["Mary"]).toarray()
print(clf.predict(vect))
"""

#---------------------------------------
#		BULK PREDICTIONS
#---------------------------------------
sample_name = ["Adam","David","Ama","Ayo","Xhavier","Ovetta","Tathiana","Xia","Joseph","Xianliang"]

# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
    	print(str(test_name) + " is Female")
    	return("Female")
        
    else:
    	print(str(test_name) + " is Male")
    	return("Male")
        

for n in sample_name:
	genderpredictor(n)



#---------------------------------------
#		SAVE MODEL
#---------------------------------------
import joblib

print('Saving Model')
NaiveBayesModel = open("models/nbGenderModel.pkl","wb")
joblib.dump(clf,NaiveBayesModel)
NaiveBayesModel.close()


vectorOut = open("models/vectorizer.pkl","wb")
joblib.dump(vectorizer, vectorOut)

