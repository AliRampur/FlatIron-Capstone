
#!pip install streamlit
#!pip install seaborn
#!pip install matplotlib

from locale import currency
from operator import index
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import pickle
import numpy as np
# import seaborn as sns
import altair as alt
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

col_names = ['step','customer', 'age', 'gender','merchant','category','amount']

uploaded_file = st.file_uploader(label='Upload Payment Transactional Detail Here. Note: Data must contain following fields - customer id, age group, gender, merchant id, category, amount, and day.', accept_multiple_files = False)

def day_func(numday):
    if (numday - 1) == 0 or (numday - 1) % 7 == 0:
        return 'Monday'
    elif (numday - 2) == 0 or (numday - 2) % 7 == 0:
        return 'Tuesday'
    elif (numday - 3) == 0 or (numday - 3) % 7 == 0:
        return 'Wednesday'
    elif (numday - 4) == 0 or (numday - 4) % 7 == 0:
        return 'Thursday'
    elif (numday - 5) == 0 or (numday - 5) % 7 == 0:
        return 'Friday'
    elif (numday - 6) == 0 or (numday - 6) % 7 == 0:
        return 'Saturday'
    elif (numday - 7) == 0 or (numday - 7) % 7 == 0:
        return 'Sunday'



if st.button(label='Click to Make Prediction'):
    if uploaded_file is not None:
        
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df['day']=uploaded_df['step'].apply(lambda x: day_func(x))

        x_sample = uploaded_df[['customer','age','gender','merchant','category','amount','day']]
        

        loaded_model = pickle.load(open("pickled models/final_logreg_fraud_model.sav", 'rb'))

        prediction = loaded_model.predict(x_sample)
        pred_proba = loaded_model.predict_proba(x_sample)


        df_prediction = pd.DataFrame(zip(prediction,pred_proba[:,1]*100),columns=['pred_fraud','Fraud Probability (%)'])
        df_prediction['Potentially Fraudulent?'] = np.where(df_prediction['pred_fraud'] == 1, 'Yes','No')

        
        df_combined = uploaded_df.join(df_prediction)        

        def count_fraud(df):
            trx_count = len(df)
            fraud_count = len(df[df['Potentially Fraudulent?'] == 'Yes'])
            total_fraudamount = df[df['Potentially Fraudulent?'] == 'Yes']['amount'].sum().round(2)
            return(f'There are {fraud_count} potentially fraudulent transactions out of {trx_count} in the uploaded dataset, totaling ${total_fraudamount}.')

        df_combined = df_combined.astype({'amount':'float'})
        df_combined['amount'] = df_combined['amount'].round(decimals=2)

        df_category = df_combined[df_combined['Potentially Fraudulent?'] == 'Yes'].groupby('category')['amount','pred_fraud'].sum().sort_values(by='amount',ascending=False).reset_index()

        df_customer = df_combined[df_combined['Potentially Fraudulent?'] == 'Yes'].groupby('customer')['amount','pred_fraud'].sum().sort_values(by='amount',ascending=False)
        df_customer = df_customer.rename(columns={'amount':'Total $','pred_fraud':'Count'})

        df_merchant = df_combined[df_combined['Potentially Fraudulent?'] == 'Yes'].groupby('merchant')['amount','pred_fraud'].sum().sort_values(by='amount',ascending=False)
        df_merchant = df_merchant.rename(columns={'amount':'Total $','pred_fraud':'Count'})

        df_days = df_combined[df_combined['Potentially Fraudulent?'] == 'Yes'].groupby('day')['amount','pred_fraud'].sum().sort_values(by='amount',ascending=False)
        df_days = df_days.rename(columns={'amount':'Total $','pred_fraud':'Count'})
        
        df_combined =df_combined.drop(columns=['step','fraud','zipMerchant','zipcodeOri','pred_fraud'])

        new_cols = ['Potentially Fraudulent?','Fraud Probability (%)','amount','customer','age','merchant','category','day']
        
        df_combined = df_combined[new_cols]

        st.subheader(count_fraud(df_combined))
        st.write('#### See table below for fraud prediction of each transaction and its associated fraud probability:')
        st.write(df_combined.sort_values(by=['Fraud Probability (%)','amount'],ascending=False))

        st.subheader('Bar Graph of Potentially Fraudulent Payments by Category (Top 5)')
        st.bar_chart(data=df_category.head(5),x='category',y='amount')

        st.subheader('Potentially Fraudulent Payments by Customer ID:')
        st.write(df_customer)

        st.subheader('Potentially Fraudulent Payments by Merchant ID:')
        st.write(df_merchant)

        st.subheader('Potentially Fraudulent Payments by Day of the Week:')
        st.write(df_days)


        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(df_combined)

        st.download_button('Click to Download Predictions into CSV',csv,'Fraud_model_results.csv','text/csv',key='download-csv')

    else:
        st.write('## You have not uploaded the necessary payment data!')
        #st.write(alt.Chart(df_combined).mark_bar().encode(x=alt.X('category',sort='x'),y='amount'))

        # def countPlot():
        #     fig = plt.figure(figsize=(10, 4))
        #     sns.countplot(x = "category", data = df_combined)
        #     st.pyplot(fig)
        
        # st.write(countPlot())

