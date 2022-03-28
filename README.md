# CREDIT-CARD-FRAUD-DETECTION-USING-ML 

❖ Find out the fraud on the transaction.

❖ Checking the fraud and non-fraud ratio.

❖ Using Random Forest Algorithm, XGBoost, Decision Tree, Logistic Regression and SVM

❖ Ul by using Python Streamlit

## Installation
    pip install streamlit
    streamlit hello
## Run the file

    streamlit run new2.py

## Objective 
1. Data understanding and exploring

3. Data cleaning

    • Handling missing values
   
   • Outliers treatment
   
3. Exploratory data analysis
   
   • Univariate analysis
   
   • Bivariate analysis
   
4. Prepare the data for modelling
     
     • Check the skewness of the data and mitigate it for fair analysis
   
   • Handling data imbalance as we see only 0.172% records are the fraud transactions
   
5. Split the data into train and test set
    
    • Scale the data (normalization)
    
6. Model building
    
    Train the model with various algorithm such as Logistic regression, SVM, Decision Tree, Random forest, XGBoost .
    
7. Model evaluation
     
     • As we see that the data is heavily imbalanced, Accuracy may not be the correct measure for this particular case
     
     • We have to look for a balance between Precision and Recall over Accuracy
     
     • We also have to find out the good ROC score with high TPR and low FPR in order to get the lower number of misclassifications.

![image](https://user-images.githubusercontent.com/88305984/160350168-39a883cc-655e-47ed-b486-39cf38375326.png)

## About Data
  
  The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, 
the positive class (frauds) account for 0.172% of all transactions.

  It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues,
we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, 
the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the
first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. 
Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
![image](https://user-images.githubusercontent.com/88305984/160350366-c1b881d5-4451-46ce-9e6b-df27913945cf.png)

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not 
meaningful for unbalanced classification.
The data for this article can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Software Used

	Python Language
    
  •	Streamlit
      
  •	Models

	Random Forest Algorithm 

	 XGBOOST 

	 Decision Tree Classification 

	 Logistic Regression 

	 SVM Classification

•	Pandas

## Streamlit
  Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. 
In just a few minutes you can build and deploy powerful data apps.
  
## Models

## Random Forest Algorithm

A random forest is a machine learning technique that's used to solve regression and classification problems. It utilizes ensemble learning. Which is a technique
that combines many classifiers to provide solutions to complex problems. A random forest algorithm consists of many decision trees.
 ![image](https://user-images.githubusercontent.com/88305984/160350051-412954ab-c519-4229-ae33-ab6a605c1e41.png)


## XGBOOST 

XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an
implementation of gradient boosted decision trees designed for speed and performance. 
![image](https://user-images.githubusercontent.com/88305984/160350010-70fe8e08-3864-403d-a68e-54ea330bdae3.png)

 
## Decision Tree Classification 
  Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value
of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
![image](https://user-images.githubusercontent.com/88305984/160349975-bf1eb2bb-5829-4ca5-aa7c-97786b6c529e.png)

 

## Logistic Regression

Logistic Regression is one of the most used ML algorithms in binary classification. Logistic Regression was used in the biological sciences in early 20th century.
It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.
![image](https://user-images.githubusercontent.com/88305984/160349928-2af28e1c-2cda-44bc-9272-cfbe927da04f.png)

 
## SVM

Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. 
However, primarily, it is used for Classification problems in Machine Learning.
![image](https://user-images.githubusercontent.com/88305984/160349877-3c3dab6e-de24-4f9b-8ab1-48f2bb75c979.png)


 
## Flow Chart
![image](https://user-images.githubusercontent.com/88305984/160346290-e6868bb7-f54c-47bb-b150-a4e47bbf2169.png)
