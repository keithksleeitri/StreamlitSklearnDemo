from typing import Literal
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score

LABEL_NAME = 'Class variable'

st.title('Diabetes')

df = pd.read_csv('diabetes.csv')
st.dataframe(df)

fig = go.Figure(data=[go.Pie(labels=['negative', 'positive'], values=[(df[LABEL_NAME] == 0).sum(), (df[LABEL_NAME] == 1).sum()])])
st.write([(df[LABEL_NAME] == 0).sum(), (df[LABEL_NAME] == 1).sum()])
st.plotly_chart(fig)

def get_model(model_type: Literal['SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest'], *arg, **kwargs) -> BaseEstimator:
    if model_type == 'SVM':
        return LinearSVC(*arg, **kwargs)
    elif model_type == 'Decision Tree':
        return DecisionTreeClassifier(*arg, **kwargs)
    elif model_type == 'Logistic Regression':
        return LogisticRegression(*arg, **kwargs)
    elif model_type == 'Random Forest':
        return RandomForestClassifier(*arg, **kwargs)

model_type: Literal['SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest'] = st.selectbox('Choose model', ['SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest'])
model = get_model(model_type)
st.write(model)

# Assuming df is your DataFrame and 'label' is the target column
features = df.drop(LABEL_NAME, axis=1)
labels = df[LABEL_NAME]

test_size = st.slider('Test Size', 0.05, 0.95, 0.2, step=0.05)

# Split the data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, random_state=42)
st.write('Training Data Size', len(labels_train))
st.write('Test Data Size', len(labels_test))

# Train the model
model.fit(features_train, labels_train)

# Now your model is ready to make predictions
predictions = model.predict(features_test)

st.text('Predict')
st.write(predictions)
st.text('Label')
st.write(labels_test)

acc = accuracy_score(labels_test, predictions)
st.metric('Accuracy', acc)
f1 = f1_score(labels_test, predictions)
st.metric('F1', f1)
precision = precision_score(labels_test, predictions)
st.metric('Precision', precision)
recall = recall_score(labels_test, predictions)
st.metric('Recall', recall)
st.write(classification_report(labels_test, predictions, output_dict=True))
# st.markdown(f"""
# '''
# {classification_report(labels_test, predictions, output_dict=False)}
# '''
# """)
