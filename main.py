import streamlit as st
import pandas as pd

st.title('Diabetes')

df = pd.read_csv('diabetes.csv')
st.dataframe(df)
