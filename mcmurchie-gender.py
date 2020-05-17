"""
#----------------------------------------------------------------------------------------
#
# Program: mcmurchie-gender.py
# Date: 15 may 2020
# Description: This is a webapp, that imports either the nb gender model
#			   or a custom library which produces a custom classification
#			   This app clasifies gender
# 
#								 
#----------------------------------------------------------------------------------------
"""
import streamlit as st 
from datetime import date
from PIL import Image
import joblib 


# def select_model(model):
# 	if model == "Standard":
# 		nb_model = open("models/nbGenderModel.pkl","rb")
# 		nb_clf = joblib.load(nb_model)
# 		answer = 'standard selected'
# 		return answer 
# 	else:
# 		nb_model = open("models/decisiontree.pkl","rb")
# 		nb_clf = joblib.load(nb_model)
# 		answer = 'advanced selected'
# 		return answer
		
# load Vectorizer For Gender Prediction
gender_vectorizer = open("models/vectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)
gender_nv_model = open("models/nbGenderModel.pkl","rb")
clf = joblib.load(gender_nv_model)



def predictGender(model, firstname):
	if model == 'Standard':
		vect = gender_cv.transform([firstname]).toarray()
		result = clf.predict(vect)
		return result
	else:
		from custom_classifier import predict 
		score = predict(firstname)
		if score == 'Female':
			result = 0
		else:
			result = 1
		return result




def main():
	st.title("AI Gender Classifier")
	st.image('images/gender.png')
	st.text("Insert your name and try it out")

	model = st.selectbox("Choose your model", ["Standard", "Advanced"])
	#predictor = predictGender(model)
	#st.info(model)
	
	firstname = st.text_input("Enter your name", "enter here")

	if st.button("Predict"):
		result = predictGender(model,firstname)
		if result == 0:
			gender = "Female"
			image = 'images/female.png'
		elif result == 1:
			gender = "Male"
			image = 'images/male.png'
		else:
			st.write('gender not defined')
			image = 'images/male.png'
		st.success('Name: {} was classified as {}'.format(firstname.title(),gender))
		img = Image.open(image)
		st.image(img,width=300)


if __name__ == '__main__':
	main()





