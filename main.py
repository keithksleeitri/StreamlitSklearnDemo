import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.title('Diabetes')

df = pd.read_csv('diabetes.csv')
st.dataframe(df)

fig = go.Figure(data=[go.Pie(labels=['negative', 'positive'], values=[(df['Class variable'] == 0).sum(), (df['Class variable'] == 1).sum()])])
st.write([(df['Class variable'] == 0).sum(), (df['Class variable'] == 1).sum()])
st.plotly_chart(fig)
