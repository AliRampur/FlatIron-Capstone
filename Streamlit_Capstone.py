
#!pip install streamlit

from operator import index
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, \
ExtraTreesClassifier, VotingClassifier, StackingRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import plot_confusion_matrix, recall_score,\
    accuracy_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.dummy import DummyClassifier



st.write('# Fraud Detection - Payment Transactions')

st.write('## Summary of Train Data')

col_names = ['step','customer', 'age', 'gender','merchant','category','amount']

uploaded_file = st.file_uploader(label='Upload Payment Transactional Detail Here. Note: Data must contain following fields - step, customer, age, gender,merchant,category, and amount', accept_multiple_files = False)


if st.button(label='Click to Make Prediction'):
    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        x_sample = uploaded_df[['step','customer','age','gender','merchant','category','amount']]
        
        st.write(x_sample)
    
        loaded_model = pickle.load(open("knn_fraud_model.sav", 'rb'))

        prediction = loaded_model.predict(x_sample)
        pred_proba = loaded_model.predict_proba(x_sample)


        df_prediction = pd.DataFrame(zip(prediction,pred_proba[:,1]*100),columns=['Fraud Prediction','Fraud Probability (%)'])
        df_prediction['Potentially Fraudulent?'] = np.where(df_prediction['Fraud Prediction'] == 1, 'Yes','No')

        
        df_combined = uploaded_df.join(df_prediction)

        st.write(df_combined.sort_values(by=['Fraud Probability (%)'],ascending=False))

