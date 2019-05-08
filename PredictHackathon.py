# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:49:09 2019

@author: himanshu.ranjan
"""
##################### import libraries #################
import pandas as pd
from HackathonFunctions import ChangeColToLabel, TransformPredictCols, ExportDfToExcel
import pickle


##################### code variables ####################
FILE_NAME = "Predict_Input.csv"
IN_FOLDER =     './In_folder/'
FEATURES_COLS = ["AgeLabel", "BusinessTravel", "DailyRatesLabel", "Department","DistanceFromHomeLabel",
               "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRateLabel",
               "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", 
               "MonthlyIncomeLabel", "MonthlyRateLabel", "NumCompaniesWorkedLabel", "OverTime", "PercentSalaryHikeLabel",
               "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYearsLabel",
               "TrainingTimesLastYearLabel", "WorkLifeBalance", "YearsAtCompanyLabel", "YearsInCurrentRoleLabel",
               "YearsSinceLastPromotionLabel", "YearsWithCurrManagerLabel"]

out_data_path  = './Out_folder/'
FILE_PATH = IN_FOLDER + FILE_NAME

############### import file to be predicted and set features columns ##########
if FILE_NAME.split(".")[1].lower() == "csv":
    rawdf = pd.read_csv(FILE_PATH)
    print("Read ", rawdf.shape[0], " rows from ", FILE_NAME)
elif FILE_NAME.split(".")[1].lower() == "xlsx":
    rawdf = pd.read_excel(FILE_PATH)
    print("Read ", rawdf.shape[0], " rows from ", FILE_NAME)
else : print("Please provide a valid file or a file extension")

initialDF = rawdf.copy()
rawdf = ChangeColToLabel(rawdf)
rawdf = rawdf[FEATURES_COLS]
rawdf = TransformPredictCols(rawdf) # load all the columns from pickles

#### load model and Attrition from pickle file #############
model_file = './Model/model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)
    
attrition_file = './Model/pkl_leAttrition.pkl'
with open(attrition_file, 'rb') as file:
    leAttrition = pickle.load(file)

######### Predict and export the predicted file #############
pred_col = model.predict(rawdf[FEATURES_COLS])
pred_col = leAttrition.inverse_transform(pred_col)
pred_col = pd.Series(pred_col)
initialDF = pd.concat([initialDF, pred_col],axis=1, sort=False )
initialDF.rename(columns = {0:"Predicted Attrition"}, inplace=True)
ExportDfToExcel(initialDF, out_data_path)
