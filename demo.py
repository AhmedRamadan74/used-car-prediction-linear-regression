import streamlit as st
import pickle 
import pandas as pd
import numpy as np
#load preprocessor and model

preprocessor=pickle.load(open("preprocessor.pkl",'rb'))
model1 = pickle.load(open("model.pkl","rb"))

# streamlit layout

st.set_page_config(page_title="Used cars price prediction",layout="wide")

st.title("Used cars price prediction, By [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/)")

st.header("About :")
st.markdown("In this web site we can predict in india if your have a car  and want to  know price it only by entry your car specifications ")
st.markdown("-----------------")

st.subheader("Car Specifications")

#input data

Brand=st.selectbox("Brand : ",['Maruti', 'Hyundai', 'Honda', 'Audi', 'Nissan', 'Toyota',
       'Volkswagen', 'Tata', 'Land', 'Mitsubishi', 'Renault',
       'Mercedes-Benz', 'BMW', 'Mahindra', 'Ford', 'Porsche', 'Datsun',
       'Jaguar', 'Volvo', 'Chevrolet', 'Skoda', 'Mini', 'Fiat', 'Jeep',
       'Smart', 'Ambassador', 'Isuzu', 'ISUZU', 'Force', 'Bentley',
       'Lamborghini'])

model=st.text_input("Enter Model of car :")

Location=st.selectbox("Enter your city :",['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Hyderabad', 'Jaipur',
       'Kochi', 'Kolkata', 'Delhi', 'Bangalore', 'Ahmedabad'])

Year=st.slider("Enter the year of car :",1998,2019)

Kilometers_Driven=st.number_input("Enter Kilometers Driven of car :")

Fuel_Type=st.selectbox("Fuel type :",['CNG', 'Diesel', 'Petrol', 'LPG', 'Electric'])

Transmission=st.selectbox("Transmission :",['Manual', 'Automatic'])

Owner_Type=st.selectbox("Owner Type :",['First', 'Second', 'Third', 'Fourth & Above'])

Mileage=st.number_input("Enter mileage of car  :")

Engine=st.number_input("Enter Engine of car  :")

Power=st.number_input("Enter Power of car  :")

Seats=st.number_input("Enter Seats of car  :")

data={"Brand":Brand,"model":model,
      "Location":Location,"Year":Year,
      "Kilometers_Driven":Kilometers_Driven,
      "Fuel_Type":Fuel_Type,"Transmission":Transmission,
      "Owner_Type":Owner_Type,"Mileage":Mileage,
      "Engine":Engine,"Power":Power,"Seats":Seats}
df=pd.DataFrame(data,index=[0])

#preproccessing

df["Owner_Type"]=df["Owner_Type"].map({"First":3,"Second" :2,"Third":1,"Fourth & Above":0})


X_test_preprocessor=preprocessor.transform(df)

# model
log_price=model1.predict(X_test_preprocessor)
price=np.exp(log_price)

price_usd=round(price[0] * 1215.19,2)
btn=st.button("Predict")

if btn:
      st.subheader("Price in Dollar")
      st.write(f"Price = {price_usd} $")
