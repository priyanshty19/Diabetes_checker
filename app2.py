import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier   
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("data//diabetes.csv")

# data.rename(columns={'pregnancies':'Times of Pregnancies'},inplace=True)
data.drop("DiabetesPedigreeFunction",axis=1,inplace=True)
data.drop("SkinThickness",axis=1,inplace=True)
x=data.drop("Outcome",axis=1)
y=data["Outcome"]
xtr,xt,ytr,yt=train_test_split(x,y,test_size=0.18,random_state=101)



page_bg_img = '''
<style>
body {
background-image: url("https://www.oogazone.com/wp-content/uploads/2018/05/top-10-fad-clipart-free-clip-art-medical-cdr.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)





st.title("Diabetes Predictor")
nav=st.sidebar.radio("Navigation",["Home","Check If Your Are Diabetic"])

if nav=="Home":

    # st.image("data//health-insurance.jpg",width=400,height=400)


    st.header("Description of Dataset")
    st.subheader("Context")
    st.write("This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.")
    st.subheader("Content")
    st.write("The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.")


    # st.subheader("Tabulated Data")
    # st.table(data)


if nav=='Check If Your Are Diabetic':
    st.header("Diabetes Check")
    val1=st.slider("Times Of Pregnancies",min_value=0,max_value=20,step=1)
    val2=st.slider("Glucose",min_value=0,max_value=200,step=1)
    val3=st.slider("Blood Pressure",min_value=0,max_value=200,step=1)
    val4=st.slider("Insulin",min_value=0,max_value=200,step=1)
    val5=st.slider("BMI",min_value=0.0,max_value=80.1,step=0.1)
    val6=st.slider("Age",min_value=0,max_value=100,step=1)
    val=[[val1,val2,val3,val4,val5,val6]]
    logmodel=LogisticRegression(random_state=101)
    logmodel.fit(xtr,ytr)
    prediction=logmodel.predict(val)    

    if st.button("Check"):
        st.success(f"You are {(prediction)}")

    st.write("1 : Diabetic")
    st.write("0 : Non-Diabetic")

