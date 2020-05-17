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




def predictGender(model):
	if model == 'Standard':
		from classify_gender import genderpredictor
		predictor = genderpredictor
		return predictor
	else:
		from custom_classifier import predict 
		predictor = predict
		return predictor






def main():
	st.title("AI Gender Classifier")
	st.image('images/gender.png')
	st.text("Insert your name and try it out")

	model = st.selectbox("Choose your model", ["Standard", "Advanced"])
	predictor = predictGender(model)
	st.info(model)
	

	firstname = st.text_input("Enter your name", "enter here")

	if st.button("Predict"):
		gender = predictor(firstname)
		if gender == "Female":
			image = 'images/female.png'
		elif gender == "Male":
			image = 'images/male.png'
		else:
			st.write('gender not defined')
			image = 'male.png'

		st.success('Name: {} was classified as {}'.format(firstname.title(),gender))
		img = Image.open(image)
		st.image(img,width=300)


if __name__ == '__main__':
	main()





