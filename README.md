# Capstone Project - Payment Fraud Detection (General Ledger)


# 1. Overview

For this project, I used classification modeling, along with pipelines, cross validation and grid searches, to create the most effective model to predict a binary class (fraud or not fraud) pertaining to payments made by customers to certain vendors in North Caroline.

   - Link to Technical Notebook: https://github.com/AliRampur/FlatIron-Capstone/blob/master/Capstone%20Notebook%20-%20Payment%20Fraud.ipynb
   - Link to final presentation: https://github.com/AliRampur/FlatIron-Capstone/blob/master/presentation.pptx
   - Link to original and sample data sources: https://github.com/AliRampur/FlatIron-Capstone/tree/master/source%20data
   - Link to Streamlit: https://alirampur-flatiron-capstone-streamlit-capstone-yx8toc.streamlitapp.com/
   
   
# 2. Business Problem

Fitris Law has an online retail client who is losing over 10% of annual revenue to fraudulent payments via its online payment portal, which is significantly higher than the industry average (5%).

They have asked my firm to develop a model that will identify potentially fraudulent payments and rank these transactions at the end of each weekly period.

The model and underlying data will allow me to do that for certain types of payment receipt systems (SAP, Oracle, venmo, zelle, paypal, etc.).


# 3. Exploratory Data Analysis and Pre-Processing Steps

In this first step, I analyzed and considered the payment data. Some of the EDA and pre-processing steps included:
- Randomly selecting 100,000 payment transactions from the total population to make the dataset more feasible
- Analyzing payment trends, including:
    - most common fraud category
    - highest average fraud category
    - average fraud payment vs non-fraud payment, by category
    - fraudulent payments by gender
    - fraudulent payments by age
    - fraudulent transaction amount by day
- Applying train-test-split to the sample data population
- Applying SMOTENC on the train dataset at 50% to improve class balance 
- Creating a pipeline for each classification model type that would contain one-hot encoding of categorical variables(gender, category, etc.) and standard scaling on continuous variables (e.g. transaction amount), including:
    - Dummy classifier
    - Logistic Regression
    - K-nearest neighbors
    - Random Forest Classifier
    - GradientBoost Classifier    



   ### Target Feature ('fraud') - Binary (fraud vs. non-fraud)
   The target feature or column is the 'fraud' column.
   
   Prior to applying SMOTENC
    - Count of Train Set: Fraudulent Payments = 874
    - Count of Train Set: Non-Fraudulent Payments = 74,126
    
    After applying SMOTENC at 50%
    - Count of Fraudulent Payments = 37,063
    - Count of Non-Fraudulent Payments = 74,126
  
  
    ### Visualizations:
   Here is a bar graph of various key attributes and features across the sample transaction population:
   
   ![image](https://github.com/AliRampur/FlatIron-Capstone/blob/master/graphs/Avg_fraud_category.png)
   
   
   
   
   ![image](https://github.com/AliRampur/FlatIron-Capstone/blob/master/graphs/Total_fraud_category.png)
   
   
   
   
   ![image](https://github.com/AliRampur/FlatIron-Capstone/blob/master/graphs/Total_Fraud_ByGender.png)
   
   
   
   
   ![image](https://github.com/AliRampur/FlatIron-Capstone/blob/master/graphs/Total_FraudbyDay.png)
   
   
   
      
# 4. Applying Cross Validation, Pipelines, and Gridsearching

As part of the modeling process, I setup a class ModelWithCV() (taken from Flatiron Lecture #51) to help streamline the process of applying cross validation and extracting the results on each classification pipeline and the given model. 

Based on the cross validation and pipeline, the most successful model for this binary prediction were:

    1. Logistic Regression: 0.99593 Accuracy Score
    2. K-nearest Neighbors: 0.933
    3. Random Forest Classification: 0.856
    



   

# 5.  Final Classification Model

After further consideration of the 3 model types identified above, I applied a grid search and the best model is Logistic Regression:

    - Y - Test Set Accuracy Score: .993


Here is the confusion matrix on the test data:

   ![image](https://github.com/AliRampur/FlatIron-Capstone/blob/master/graphs/Confusion%20Matrix.png)
   
   

# 6.  Streamlit - Uploading Payment Transactions

You can open the application and upload any set of similar transactional data:

https://alirampur-flatiron-capstone-streamlit-capstone-yx8toc.streamlitapp.com/
   
The application will then indicate which transactions are potentially fraudulent and the probability associated with each transaction.

Some additional summary graphs and metrics are also provided to help the user assess which procedures and controls may need to be modified based on the results (i.e. increase oversight on certain days of the week, certain customers have a large number of fraudulent transactions, etc.).


# 7. Recommendations / Next Steps


Based on the results of the final model, here are my recommendations:
    
    1. At the end of each weekly or two-week period, feed the Final Logistic Regression model with payment data.
    
    2. I can update the model to include additional locations (i.e. zipcode).
    
    3. Investigate the identified transactions and prevent delivery of any product or service unless payment has been made.





