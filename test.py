import pandas as pd
import numpy as np
import joblib

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


# load Vectorizer For Gender Prediction
gender_vectorizer = open("models/DCvectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("models/decisiontree.pkl","rb")
clf = joblib.load(gender_nv_model)


firstname = ['adam']

result = clf.predict(['adam'])
print(result)

