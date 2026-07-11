import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
st.set_page_config(page_title='Life Insurance Prediction',page_icon='🛡️')
st.title('🛡️ Life Insurance Prediction using Logistic Regression')
df=pd.read_csv('insurance_data.csv')
st.dataframe(df)
X=df[['age']]; y=df['bought_insurance']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression().fit(X_train,y_train)
st.metric('Accuracy',f"{accuracy_score(y_test,model.predict(X_test))*100:.2f}%")
fig,ax=plt.subplots()
ax.scatter(df['age'],df['bought_insurance'])
ages=sorted(df['age'])
ax.plot(ages,model.predict_proba(pd.DataFrame({'age':ages}))[:,1])
st.pyplot(fig)
age=st.slider("Age", 1, 100, 25)
if st.button('Predict'):
 p=model.predict_proba([[age]])[0][1]
 st.write(f'Probability: {p*100:.2f}%')
 st.write('Likely to Buy' if model.predict([[age]])[0] else 'Not Likely to Buy')
st.write('Confusion Matrix')
st.write(confusion_matrix(y_test,model.predict(X_test)))
