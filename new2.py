import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# title of the app
st.title("Credit Card Fraud Detection ")

#Load the Data
data=pd.read_csv("C:/Users/Sajan/Downloads/sem 2/creditcard/sample.csv")


content = st.sidebar.selectbox(
    label="Select the content",
    options=['', 'EDA', 'PLOT'])
st.header((content))

if content == 'EDA':
    Analysis = st.sidebar.selectbox(
    label = "Select",
    options = ["","Information","Head","Tail","Dimension","Description","Data Cleaning"])
    st.header((Analysis))
    
    if Analysis == "Information":
        try:
            st.subheader("Exploring Data")
            st.write("►  The dataset is It contains two-day transactions made on 09/2013 by European cardholders.")
            st.write("►  The dataset contains 492 frauds out of 284,807 transactions.")
            st.write("►  Thus, it is highly unbalanced, with the positive (frauds) accounting for only 0.17%.")
            st.write("►  Features V1, V2, … V28 are the principal components obtained with PCA transformation.")
            st.write("►  The only features which have not been transformed are ‘Time’ and ‘Amount’.")
            st.write("► ‘Time’ is the seconds elapsed between each transaction and the first. ")
            st.write("► ‘Amount’ is the transaction amount. ‘Class’ is the response variable with 1 as fraud and 0 otherwise.") 
        except Exception as e:
            print(e)
			
    if Analysis == "Head":
        try:
            st.write("It shows the First 5 rows of data")
            head = data.head()
            st.write(head)
        except Exception as e:
            print(e)
			
    if Analysis == "Tail":
        try:
            st.write("It shows the last 5 rows of data")
            tail = data.tail()
            st.write(tail)
        except Exception as e:
            print(e)
			
    if Analysis == "Dimension":
        try:
            st.write("It shows the dimension of the data")
            shape = data.shape
            st.write(shape)
        except Exception as e:
            print(e)
            
    if Analysis == "Description":
        try:
            st.write("It shows the description of the data")
            describe = data.describe()
            st.write(describe)
        except Exception as e:
            print(e)
        
    if Analysis == "Data Cleaning":
        try:
            st.write("Null value Handling")
            null = data_missing_columns = (round(((data.isnull().sum()/len(data.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
            st.write(null)
        except Exception as e:
            print(e)
			
			#############################################################





elif content == "PLOT":
	chart = st.sidebar.selectbox(
	label = "Select",
	options = ['',"Bar Plot","Distribution plot_1","Distribution plot_2"])
	st.header((chart))
	
	if chart == "Bar Plot":
		try:
			classes = data['Class'].value_counts()
			normal_share = round((classes[0]/data['Class'].count()*100),2)
			fraud_share = round((classes[1]/data['Class'].count()*100),2)
			st.write('Percentage of fraudulent vs non-fraudulent transcations')
			fig, ax = plt.subplots(figsize = (10,5))
			fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
			df_fraud_percentage = pd.DataFrame(fraud_percentage) 
			sns.barplot(x='Class',y='Percentage', data=df_fraud_percentage)
			st.write(fig)
		except Exception as e:
			print(e)
			
	if chart == "Distribution plot_1":
		try:
			# Creating fraudulent dataframe
			data_fraud = data[data['Class'] == 1]
			# Creating non fraudulent dataframe
			data_non_fraud = data[data['Class'] == 0]
			st.write('Seconds elapsed between the transction and the first transction')
			fig, ax = plt.subplots(figsize = (10,5))
			sns.distplot(data_fraud['Time'],label='fraudulent',hist=False)
			sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
			st.write(fig)
		except Exception as e:
			print(e)
			
	if chart == "Distribution plot_2":
		try:
			# Creating fraudulent dataframe
			data_fraud = data[data['Class'] == 1]
			# Creating non fraudulent dataframe
			data_non_fraud = data[data['Class'] == 0]
			# Dropping the Time column
			data.drop('Time', axis=1, inplace=True)
			st.write('Transction Amount')
			fig, ax = plt.subplots(figsize = (10,5))
			ax = sns.distplot(data_fraud['Amount'],label='fraudulent',hist=False)
			ax = sns.distplot(data_non_fraud['Time'],label='non fraudulent',hist=False)
			st.write(fig)
		except Exception as e:
			print(e)
			#################################



Algorithms = st.sidebar.selectbox(
    label="Algorithms",
    options=['', 'Random Forest', 'XGBOOST','Decision Tree','Logistic Regression','SVM Classification'])
st.header((Algorithms))


if Algorithms == 'Random Forest':
	Class = st.sidebar.selectbox(
	label = "Select",
	options = ["","Classifcation report","Accuracy Score","Confusion Matrix","Plot"])
	st.header((Class))


	if Class == "Classifcation report":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			st.subheader("Random Forest Algorithms")
			cla=(classification_report(y_test, y_pred))
			st.write(cla)   
		except Exception as e:
			print(e)
	
	
	if Class == "Accuracy Score":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			b=accuracy_score(y_test,y_pred)
			st.write(b)
		except Exception as e:
			print(e)
			
	if Class == "Confusion Matrix":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			m = confusion_matrix(y_true=y_test, y_pred=y_pred)
			st.write(m)
		except Exception as e:
			print(e)
			
	if Class == "Plot":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)
			st.write("Random Forest Algorithm")
			fig, ax = plt.subplots(figsize = (10,5))
			LABELS = ['Normal', 'Fraud'] 
			conf_matrix = confusion_matrix(y_test, y_pred) 
			sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
			st.write(fig)
		except Exception as e:
			print(e)
 
 #####################################

			
elif Algorithms == 'XGBOOST':
	XGB = st.sidebar.selectbox(
	label = "Select",
	options = ["","Classifcation report","Accuracy Score","Confusion Matrix","Plot"])
	st.header((XGB))			

	if XGB == "Classifcation report":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			
			xg = xgb.XGBClassifier()
			xg.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]

			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred

			cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)
			
			X = classification_report(y_test, pred)
			st.write(X)
			
		except Exception as e:
			print(e)

	if XGB == "Accuracy Score":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			
			xg = xgb.XGBClassifier()
			xg.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]

			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred

			cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)
			
			Z=accuracy_score(y_test, pred)
			st.write(Z)
			
		except Exception as e:
			print(e)

	if XGB == "Confusion Matrix":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			
			xg = xgb.XGBClassifier()
			xg.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]

			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred

			cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)
			
			 
			conf_matrix = confusion_matrix(y_test, pred) 
			st.write(conf_matrix)
			
		except Exception as e:
			print(e)
			
	if XGB == "Plot":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			
			xg = xgb.XGBClassifier()
			xg.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]

			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred

			cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)
			st.write("XGBOOST Algorithm")
			fig, ax = plt.subplots(figsize = (15,5))
			LABELS = ['Normal', 'Fraud'] 
			conf_matrix = confusion_matrix(y_test, pred) 
			sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
			st.write(fig)
			
		except Exception as e:
			print(e)
              #################################


elif Algorithms == 'Decision Tree':
	DETS = st.sidebar.selectbox(
	label = "Select",
	options = ["","Classifcation report","Accuracy Score","Confusion Matrix","Plot"])
	st.header((DETS))


	if DETS == "Classifcation report":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier=DecisionTreeClassifier(max_depth=4)
			classifier.fit(X_train,y_train)
			x_pred=classifier.predict(X_test)

			DT=classification_report(y_test, x_pred)
			st.write(DT)
			   
		except Exception as e:
			print(e)
	
	
	if DETS == "Accuracy Score":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier=DecisionTreeClassifier(max_depth=4)
			classifier.fit(X_train,y_train)
			x_pred=classifier.predict(X_test)
			
			DTS = accuracy_score(y_test,x_pred) 
			st.write(DTS)
			
		except Exception as e:
			print(e)
			
	if DETS == "Confusion Matrix":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			classifier=DecisionTreeClassifier(max_depth=4)
			classifier.fit(X_train,y_train)
			x_pred=classifier.predict(X_test)
			
			conf = confusion_matrix(y_true=y_test, y_pred=x_pred)
			st.write(conf)
			
		except Exception as e:
			print(e)
			
	if DETS == "Plot":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			
			classifier=DecisionTreeClassifier(max_depth=4)
			classifier.fit(X_train,y_train)
			x_pred=classifier.predict(X_test)
			
			fig, ax = plt.subplots(figsize = (10,5))
			LABELS = ['Normal', 'Fraud'] 
			conf_matrix = confusion_matrix(y_test, x_pred) 
			sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
			st.write(fig)
		except Exception as e:
			print(e)
##########################################

elif Algorithms == 'Logistic Regression':
	LR = st.sidebar.selectbox(
	label = "Select",
	options = ["","Classifcation report","Accuracy Score","Confusion Matrix","Plot"])
	st.header((LR))

	if LR == "Classifcation report":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			lr = LogisticRegression()
			lr.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]
			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred
			cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
			CL=classification_report(y_test, pred)
			st.write(CL)  
		except Exception as e:
			print(e)
	
	
	if LR == "Accuracy Score":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			lr = LogisticRegression()
			lr.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]
			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred
			cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
			AS=accuracy_score(y_test, pred)
			st.write(AS)	
		except Exception as e:
			print(e)
			
	if LR == "Confusion Matrix":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			lr = LogisticRegression()
			lr.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]
			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred
			cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
			CM = confusion_matrix(y_test, pred) 
			st.write(CM)	
		except Exception as e:
			print(e)
			
	if LR == "Plot":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
			lr = LogisticRegression()
			lr.fit(X_train, y_train)
			def PrintStats(cmat, y_test, pred):
				tpos = cmat[0][0]
				fneg = cmat[1][1]
				fpos = cmat[0][1]
				tneg = cmat[1][0]
			def RunModel(model, X_train, y_train, X_test, y_test):
				model.fit(X_train, y_train.values.ravel())
				pred = model.predict(X_test)
				matrix = confusion_matrix(y_test, pred)
				return matrix, pred
			cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
			fig, ax = plt.subplots(figsize = (8,4))
			LABELS = ['Normal', 'Fraud'] 
			conf_matrix = confusion_matrix(y_test, pred)  
			sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
			st.write(fig)
		except Exception as e:
			print(e)
			#############################
elif Algorithms == 'SVM Classification':
	SVM = st.sidebar.selectbox(
	label = "Select",
	options = ["","Classifcation report","Accuracy Score","Confusion Matrix","Plot"])
	st.header((SVM))
	
	if SVM == "Classifcation report":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
		
			clf = svm.SVC()
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
		
			CR=classification_report(y_test,predictions)
			st.write(CR)
		
		except Exception as e:
			print(e)
	
	
	if SVM == "Accuracy Score":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
		
			clf = svm.SVC()
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
		
			AS=accuracy_score(y_test, predictions)
			st.write(AS)
		
		except Exception as e:
			print(e)
			
	if SVM == "Confusion Matrix":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
		
			clf = svm.SVC()
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
		
			CM=confusion_matrix(y_test,predictions)
			st.write(CM)
		except Exception as e:
			print(e)
			
	if SVM == "Plot":
		try:
			y = data['Class']
			X = data.drop(['Class'], axis=1)
			X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
		
			clf = svm.SVC()
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_test)
		
			fig, ax = plt.subplots(figsize = (8,4))
			LABELS = ['Normal', 'Fraud'] 
			conf_matrix = confusion_matrix(y_test,predictions)  
			sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
			st.write(fig)
		except Exception as e:
			print(e)