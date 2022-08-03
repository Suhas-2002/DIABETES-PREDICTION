import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv("diabetes.csv")
X = data.drop(['Outcome'],axis='columns')
Y = data['Outcome']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rdc = RandomForestClassifier()
rdc.fit(x_train,y_train)

st.write("""# Diabetes Predictor
This app predicts whether you have **Diabetes or not**!""")

st.write('<div style="text-align: right;"> ~Suhas Prabhu </div>', unsafe_allow_html=True)
st.write('---')

preg = st.number_input('Pregnencies :', min_value=0) 
glu = st.number_input('Glucose : ', min_value=0) 
bp = st.number_input('Blood Pressure : ', min_value=0) 
skt = st.number_input('Skin Thickness : ', min_value=0) 
ins = st.number_input('Insulin : ', min_value=0) 
bmi = st.number_input('BMI : ', min_value=0.0) 
dpf = st.number_input('Diabetes Pedegree : ', min_value=0.0) 
age = st.number_input('Age : ', min_value=0) 

def check_input():
    if (not preg or not glu or not bp or not skt or not ins or not bmi or not dpf or not age): 
        st.warning("Please enter all the fields!")
        return False

    else : return True

df = pd.DataFrame([[preg,glu,bp,skt,ins,bmi,dpf,age]],
              columns=['Pregnencies','Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedegree','Age'])
input = [[preg,glu,bp,skt,ins,bmi,dpf,age]]

st.header('Specified Input parameters')
st.write(df)
st.write('---')


pred=""

submit = st.button('Predict')



def print_results():
    output = rdc.predict(sc.transform(input))
    if((str(output)[1:-1])=='0'):
        pred = "Congratulations! You don't have diabetes."
    else:
        pred = "Our evaluation system has discovered that you most likely have diabetes. We suggest you visit a doctor."
    
    st.write("""### """ ,pred)  

if submit:
    if check_input():
        print_results()

