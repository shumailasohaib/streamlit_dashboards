import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from numpy.core.numeric import True_
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


#Heading
df= pd.read_csv("diabetes.csv")
st.title("Diabetes prediction app")
st.sidebar.header("Paitent Data")
st.subheader("Description of stats of data")
st.write(df.describe())
#Split the data into Train Test split
X=df.drop(["Outcome"],axis =1)
y=df.iloc[:,-1]



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Function
def user_report():
    pregnancies=st.sidebar.slider("Pregnancies",0,17,2)
    glucose= st.sidebar.slider("Glucose",0,199,110)
    bp=st.sidebar.slider("BloodPressure",0,122,80)
    sk=st.sidebar.slider("SkinThickness",0,99,12)
    insulin=st.sidebar.slider("Insulin",0,846,80)
    bmi=st.sidebar.slider("BMI",0,67,5)
    dpf= st.sidebar.slider("DiabetesPedigreeFunction",0.07,2.42,0.37)
    age=st.sidebar.slider("Age",21,81,33)
    user_report_data={
        "pregnancies":pregnancies,
        "glucose": glucose,
        "bp":bp,
        "sk":sk,
        "insulin":insulin,
        "bmi":bmi,
        "dpf":dpf,
        "age":age}
    report_data=pd.DataFrame(user_report_data,index=[0])
    return report_data

user_data=user_report()
st.subheader("Patient data")
st.write("user_data")

#Model
rc=RandomForestClassifier()
rc.fit(X_train,y_train)
user_result=rc.predict(user_data)

#Visulization
st.title("Visualized patient Data")

#color Function
if user_result[0]==0:
    color = "blue"
else:
    color = "red"
    
# Age vs pregnancies
st.header("Pregnancy count graph (other vs yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x="Age", y = "Pregnancies",data= df,hue = "Outcome")
ax2 = sns.scatterplot(x=user_data["age"],y=user_data["pregnancies"], s=150 , color = color)
plt.xticks(np.arange(10,100,50))
plt.yticks(np.arange(0,20,2))
plt.title("0-Healthy & 1 - Diabetic")
st.pyplot(fig_preg) 

# Age vs Bloodpressure
st.header("BloodPressure count graph (other vs yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x="Age", y = "BloodPressure",data= df,hue ="Outcome")
ax2 = sns.scatterplot(x=user_data["age"],y=user_data["bp"], s=150 , color = color)
plt.xticks(np.arange(10,100,50))
plt.yticks(np.arange(0,20,2))
plt.title("0-Healthy & 1 - Diabetic")
st.pyplot(fig_preg) 

# Age vs SkinThickness
st.header("BloodPressure count graph (other vs yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x="Age", y = "SkinThickness",data= df,hue ="Outcome")
ax2 = sns.scatterplot(x=user_data["age"],y=user_data["sk"], s=150 , color = color)
plt.xticks(np.arange(10,100,50))
plt.yticks(np.arange(0,20,2))
plt.title("0-Healthy & 1 - Diabetic")
st.pyplot(fig_preg) 


# pregnancies Vs Glucose
st.header("Glucose count graph (other vs yours)")
fig_preg = plt.figure()
ax1 = sns.scatterplot(x="Pregnancies", y = "Glucose",data= df,hue = "Outcome")
ax2 = sns.scatterplot(x=user_data["pregnancies"],y=user_data["glucose"], s=150 , color = color)
plt.xticks(np.arange(10,100,50))
plt.yticks(np.arange(0,20,2))
plt.title("0-Healthy & 1 - Diabetic")
st.pyplot(fig_preg) 




# Output
st.header("your Report:")  
output = " "  
if user_result[0]==0:
    output = "Your are healthy"    
    st.balloons()
else:
    output="you are Diabetic "
    st.warning("Sugger,Sugger,Sugger")
st.title(output)
    
#accuracy, recall, precision and confusion matrix
rc.fit(X_train, y_train)
accuracy = rc.score(X_test, y_test)
y_pred = rc.predict(X_test)
st.write("Accuracy :", accuracy.round(2))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall: ",recall_score(y_test, y_pred))
st.write("confision_matrix:",precision_score(y_test, y_pred))

#plotting
# Bar chart
st.subheader("Bar_chart")
st.bar_chart(df["Age"])

# Line chart
st.subheader("Line_chart")
st.line_chart(df["Pregnancies"])
    
