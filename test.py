import streamlit as st 
from datetime import date
from PIL import Image
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
import joblib 



# load Model For Gender Prediction
gender_nv_model = open("models/nbGenderModel.pkl","rb")
gender_clf = joblib.load(gender_nv_model)


cv = CountVectorizer()
vector = cv.transform('Adam').toarray()
print('vector is')
print(vector)




