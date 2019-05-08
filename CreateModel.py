# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:36:02 2019

@author: himanshu.ranjan
"""

######################## Import libraries ########################
import datetime
start = datetime.datetime.now()
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from HackathonFunctions import BalanceData, ChangeColToLabel, labelEncode

######################## Code Dictionaries ########################
USE_COLS = ["AgeLabel", "BusinessTravel", "DailyRatesLabel", "Department","DistanceFromHomeLabel",
               "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRateLabel",
               "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", 
               "MonthlyIncomeLabel", "MonthlyRateLabel", "NumCompaniesWorkedLabel", "OverTime", "PercentSalaryHikeLabel",
               "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYearsLabel",
               "TrainingTimesLastYearLabel", "WorkLifeBalance", "YearsAtCompanyLabel", "YearsInCurrentRoleLabel",
               "YearsSinceLastPromotionLabel", "YearsWithCurrManagerLabel", "Attrition"]

X_COLS = ["AgeLabel", "BusinessTravel", "DailyRatesLabel", "Department","DistanceFromHomeLabel",
               "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRateLabel",
               "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", 
               "MonthlyIncomeLabel", "MonthlyRateLabel", "NumCompaniesWorkedLabel", "OverTime", "PercentSalaryHikeLabel",
               "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYearsLabel",
               "TrainingTimesLastYearLabel", "WorkLifeBalance", "YearsAtCompanyLabel", "YearsInCurrentRoleLabel",
               "YearsSinceLastPromotionLabel", "YearsWithCurrManagerLabel"]

Y_COLS = "Attrition"
######################## Main program ########################

FILE_NAME = "NWMHackathonAIDataset.csv"
EXT = FILE_NAME.split(".")[1]

if EXT.lower() == "xlsx":
    print("Reading the file ", FILE_NAME)
    rawdf = pd.read_excel(FILE_NAME)
    print(FILE_NAME, " has been read with ", rawdf.shape[0], " lines.")
elif EXT.lower() == "csv":
    print("Reading the file ", FILE_NAME)
    rawdf = pd.read_csv(FILE_NAME)
    print(FILE_NAME, " has been read with ",rawdf.shape[0], " lines.")
else:
    print("Please provide a suitable file and file extension!!!")
    exit()

################ since this is a imbalanced dataset we are dropping Attrition=No randomly and adding addtion Attrition = Yes ################
rawdf = BalanceData(rawdf)
################ changing data columns in label columns as done below################
rawdf = ChangeColToLabel(rawdf)
################ taking dataframe with relevant columns ################
rawdf = rawdf[USE_COLS]
#######################Label Encode all columns######################################
rawdf = labelEncode(rawdf)
################################## Creating a train and test set for model ################################

X = rawdf[X_COLS]
y = rawdf[Y_COLS]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
clf = RandomForestClassifier(n_estimators=1111,bootstrap=False,max_features='auto', max_depth=90, min_samples_leaf=10, min_samples_split=5, random_state=46)
clf.fit(x_train, y_train)

######################## get test accuracy ########################
y_pred_test_rf = clf.predict(x_test)
print("Test Accuracy with Random Forest Classifier is : ", accuracy_score(y_test, y_pred_test_rf), " in ", datetime.datetime.now()- start, " time")
print('\n')
cm_test = confusion_matrix(y_test, y_pred_test_rf)
df_feat = pd.DataFrame()
df_feat["Feature"] = X_COLS
df_feat["Importance"] = clf.feature_importances_

############################### getting precision and recall #########################################
print("Test Recall is : ", np.diag(cm_test)/np.sum(cm_test, axis=0))
print('\n')

print("Test Precision is : ", np.diag(cm_test)/np.sum(cm_test, axis=1))
print('\n')

############################### saving the model and label encoder ##################################

pkl_Model = './Model/model.pkl'
with open(pkl_Model, 'wb') as file:
    pickle.dump(clf, file)
    

"""
########## trying to find the best parameter #############

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
"""