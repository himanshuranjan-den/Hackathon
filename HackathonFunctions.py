# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:04:41 2019

@author: himanshu.ranjan
"""


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def ExportDfToExcel(df, out_date_path):
    fname = out_date_path +'/Attrition_predicted.xlsx'
    print("Exporting the file to", fname, " file")
    writer = pd.ExcelWriter(fname)
    df.to_excel(writer,sheet_name='Main', index=False)
    writer.save()
    print("Exported the file with rows count: ", df.shape[0])

def BalanceData(rawdf):
    rawdf = rawdf.drop(rawdf[rawdf["Attrition"] == 'No'].sample(frac=.5).index).reset_index(drop = True)
    dfyes = rawdf[rawdf["Attrition"] == "Yes"]
    dfyes = pd.concat([dfyes]*2, axis=0).reset_index(drop = True)
    rawdf = pd.concat([rawdf, dfyes], axis=0).reset_index(drop = True)
    return rawdf

def ChangeColToLabel(rawdf):
    rawdf["DailyRatesLabel"] = rawdf["DailyRate"].apply(lambda x: "DR1" if x in np.arange(0,250) else ("DR2" if x in np.arange(250,500) else ("DR3" if x in np.arange(500,100) else "DR4")))                   
    rawdf["HourlyRateLabel"] = rawdf["HourlyRate"].apply(lambda x: "HR1" if x in np.arange(0,50) else "HR2")                   
    rawdf["MonthlyIncomeLabel"]= rawdf["MonthlyIncome"].apply(lambda x : "MI1" if x in np.arange(0,5000) else ("MI2" if x in np.arange(5000,10000) else ("MI3" if x in np.arange(10000, 15000) else "MI4")))
    rawdf["MonthlyRateLabel"]= rawdf["MonthlyRate"].apply(lambda x : "MR1" if x in np.arange(0,5000) else ("MR2" if x in np.arange(5000,10000) else ("MR3" if x in np.arange(10000, 15000) else "MR4")))
    rawdf["NumCompaniesWorkedLabel"] = rawdf["NumCompaniesWorked"].apply(lambda x: "NCW1" if x in np.arange(0,4) else ("NCW2" if x in np.arange(4,8) else "NCW3"))
    #rawdf["StockOptionLevel"] = rawdf["StockOptionLevel"] +1
    rawdf["PercentSalaryHikeLabel"] = rawdf["PercentSalaryHike"].apply(lambda x : "PH1" if x in np.arange(0,16) else ("PH2" if x in np.arange(16,20) else "PH3"))
    rawdf["TotalWorkingYearsLabel"] = rawdf["TotalWorkingYears"].apply(lambda x: "WY1" if x in np.arange(0,10) else ("WY2" if x in np.arange(10,20) else ("WY3" if x in np.arange(20, 30) else "WY4")))
    rawdf["TrainingTimesLastYearLabel"] = rawdf["TrainingTimesLastYear"].apply(lambda x : "TL1" if x in np.arange(0,3) else ("TL2" if x in np.arange(3,5) else "TL3"))
    rawdf["YearsAtCompanyLabel"] = rawdf["YearsAtCompany"].apply(lambda x: "YC1" if x in np.arange(0,11) else ("YC2" if x in np.arange(11,21) else "YC3"))
    rawdf["YearsInCurrentRoleLabel"] = rawdf["YearsInCurrentRole"].apply(lambda x: "YCR1" if x in np.arange(0,5) else ("YCR2" if x in np.arange(5,10) else ("YCR3" if x in np.arange(10.15) else "YCR4")))
    rawdf["YearsSinceLastPromotionLabel"] = rawdf["YearsSinceLastPromotion"].apply(lambda x: "YSP1" if x in np.arange(0,5) else ("YSP2" if x in np.arange(5,10) else "YSP3"))
    rawdf["YearsWithCurrManagerLabel"] = rawdf["YearsWithCurrManager"].apply(lambda x: "YCM1" if x in np.arange(0,5) else ("YCM2" if x in np.arange(5,10) else "YCM3"))
    rawdf["DistanceFromHomeLabel"] = rawdf["DistanceFromHome"].apply(lambda x: "DFH1" if x in np.arange(0,10) else ("DFH2" if x in np.arange(10, 20) else "DFH3"))
    rawdf["AgeLabel"] = rawdf["Age"].apply(lambda x: "A1" if x in np.arange(0,31) else ("A2" if x in np.arange(31,41) else ("A3" if x in np.arange(41, 51) else "A4")))
    return rawdf

def labelEncode(rawdf):
    leAge = LabelEncoder()
    rawdf["AgeLabel"] = leAge.fit_transform(rawdf["AgeLabel"])
    pkl_leAge = './Model/pkl_leAgeLabel.pkl'
    with open(pkl_leAge, 'wb') as file:
        pickle.dump(leAge, file)
    
    leBusinessTravel = LabelEncoder()
    rawdf["BusinessTravel"] = leBusinessTravel.fit_transform(rawdf["BusinessTravel"])
    pkl_leBusinessTravel = './Model/pkl_leBusinessTravel.pkl'
    with open(pkl_leBusinessTravel, 'wb') as file:
        pickle.dump(leBusinessTravel, file)
        
    leDailyRates = LabelEncoder()
    rawdf["DailyRatesLabel"] = leDailyRates.fit_transform(rawdf["DailyRatesLabel"])
    pkl_leDailyRates = './Model/pkl_leDailyRatesLabel.pkl'
    with open(pkl_leDailyRates, 'wb') as file:
        pickle.dump(leDailyRates, file)
        
    leDepartment = LabelEncoder()
    rawdf["Department"] = leDepartment.fit_transform(rawdf["Department"])
    pkl_leDepartment = './Model/pkl_leDepartment.pkl'
    with open(pkl_leDepartment, 'wb') as file:
        pickle.dump(leDepartment, file)
        
    leDistanceFromHome = LabelEncoder()
    rawdf["DistanceFromHomeLabel"] = leDistanceFromHome.fit_transform(rawdf["DistanceFromHomeLabel"])
    pkl_leDistanceFromHome = './Model/pkl_leDistanceFromHomeLabel.pkl'
    with open(pkl_leDistanceFromHome, 'wb') as file:
        pickle.dump(leDistanceFromHome, file)
    """
    leEducation = LabelEncoder()
    rawdf["Education"] = leEducation.fit_transform(rawdf["Education"])
    pkl_leEducation = './Model/pkl_leEducation.pkl'
    with open(pkl_leEducation, 'wb') as file:
        pickle.dump(leEducation, file)
    """    
    leEducationField = LabelEncoder()
    rawdf["EducationField"] = leEducationField.fit_transform(rawdf["EducationField"])
    pkl_leEducationField = './Model/pkl_leEducationField.pkl'
    with open(pkl_leEducationField, 'wb') as file:
        pickle.dump(leEducationField, file)
    
    leEnvironmentSatisfaction = LabelEncoder()
    rawdf["EnvironmentSatisfaction"] = leEnvironmentSatisfaction.fit_transform(rawdf["EnvironmentSatisfaction"])
    pkl_leEnvironmentSatisfaction = './Model/pkl_leEnvironmentSatisfaction.pkl'
    with open(pkl_leEnvironmentSatisfaction, 'wb') as file:
        pickle.dump(leEnvironmentSatisfaction, file)
        
    leGender = LabelEncoder()
    rawdf["Gender"] = leGender.fit_transform(rawdf["Gender"])
    pkl_leGender = './Model/pkl_leGender.pkl'
    with open(pkl_leGender, 'wb') as file:
        pickle.dump(leGender, file)
    
    leHourlyRate = LabelEncoder()
    rawdf["HourlyRateLabel"] = leHourlyRate.fit_transform(rawdf["HourlyRateLabel"])
    pkl_leHourlyRate = './Model/pkl_leHourlyRateLabel.pkl'
    with open(pkl_leHourlyRate, 'wb') as file:
        pickle.dump(leHourlyRate, file)
    
    leJobRole = LabelEncoder()
    rawdf["JobRole"] = leJobRole.fit_transform(rawdf["JobRole"])
    pkl_leJobRole = './Model/pkl_leJobRole.pkl'
    with open(pkl_leJobRole, 'wb') as file:
        pickle.dump(leJobRole, file)
    
    leMaritalStatus = LabelEncoder()
    rawdf["MaritalStatus"] = leMaritalStatus.fit_transform(rawdf["MaritalStatus"])
    pkl_leMaritalStatus = './Model/pkl_leMaritalStatus.pkl'
    with open(pkl_leMaritalStatus, 'wb') as file:
        pickle.dump(leMaritalStatus, file)
    
    leMonthlyIncome = LabelEncoder()
    rawdf["MonthlyIncomeLabel"] = leMonthlyIncome.fit_transform(rawdf["MonthlyIncomeLabel"])
    pkl_leMonthlyIncome = './Model/pkl_leMonthlyIncomeLabel.pkl'
    with open(pkl_leMonthlyIncome, 'wb') as file:
        pickle.dump(leMonthlyIncome, file)
        
    leMonthlyRate = LabelEncoder()
    rawdf["MonthlyRateLabel"] = leMonthlyRate.fit_transform(rawdf["MonthlyRateLabel"])
    pkl_leMonthlyRate = './Model/pkl_leMonthlyRateLabel.pkl'
    with open(pkl_leMonthlyRate, 'wb') as file:
        pickle.dump(leMonthlyRate, file)
        
    leNumCompaniesWorked = LabelEncoder()
    rawdf["NumCompaniesWorkedLabel"] = leNumCompaniesWorked.fit_transform(rawdf["NumCompaniesWorkedLabel"])
    pkl_leNumCompaniesWorked = './Model/pkl_leNumCompaniesWorkedLabel.pkl'
    with open(pkl_leNumCompaniesWorked, 'wb') as file:
        pickle.dump(leNumCompaniesWorked, file)
        
    leOverTime = LabelEncoder()
    rawdf["OverTime"] = leOverTime.fit_transform(rawdf['OverTime'])
    pkl_leOverTime = './Model/pkl_leOverTime.pkl'
    with open(pkl_leOverTime, 'wb') as file:
        pickle.dump(leOverTime, file)
        
    lePercentSalaryHike =  LabelEncoder()
    rawdf["PercentSalaryHikeLabel"] = lePercentSalaryHike.fit_transform(rawdf["PercentSalaryHikeLabel"])
    pkl_lePercentSalaryHike = './Model/pkl_lePercentSalaryHikeLabel.pkl'
    with open(pkl_lePercentSalaryHike, 'wb') as file:
        pickle.dump(lePercentSalaryHike, file)    
    
    leTotalWorkingYears = LabelEncoder()
    rawdf["TotalWorkingYearsLabel"] = leTotalWorkingYears.fit_transform(rawdf["TotalWorkingYearsLabel"])
    pkl_leTotalWorkingYears = './Model/pkl_leTotalWorkingYearsLabel.pkl'
    with open(pkl_leTotalWorkingYears, 'wb') as file:
        pickle.dump(leTotalWorkingYears, file) 
        
    leTrainingTimesLastYear = LabelEncoder()
    rawdf["TrainingTimesLastYearLabel"] = leTrainingTimesLastYear.fit_transform(rawdf["TrainingTimesLastYearLabel"])
    pkl_leTrainingTimesLastYear = './Model/pkl_leTrainingTimesLastYearLabel.pkl'
    with open(pkl_leTrainingTimesLastYear, 'wb') as file:
        pickle.dump(leTrainingTimesLastYear, file)
        
    leYearsAtCompany = LabelEncoder()
    rawdf["YearsAtCompanyLabel"] = leYearsAtCompany.fit_transform(rawdf["YearsAtCompanyLabel"])
    pkl_leYearsAtCompany = './Model/pkl_leYearsAtCompanyLabel.pkl'
    with open(pkl_leYearsAtCompany, 'wb') as file:
        pickle.dump(leYearsAtCompany, file)
        
    leYearsInCurrentRole = LabelEncoder()
    rawdf["YearsInCurrentRoleLabel"] = leYearsInCurrentRole.fit_transform(rawdf["YearsInCurrentRoleLabel"])
    pkl_leYearsInCurrentRole = './Model/pkl_leYearsInCurrentRoleLabel.pkl'
    with open(pkl_leYearsInCurrentRole, 'wb') as file:
        pickle.dump(leYearsInCurrentRole, file)
        
    leYearsSinceLastPromotion = LabelEncoder()
    rawdf["YearsSinceLastPromotionLabel"] = leYearsSinceLastPromotion.fit_transform(rawdf["YearsSinceLastPromotionLabel"])
    pkl_leYearsSinceLastPromotion = './Model/pkl_leYearsSinceLastPromotionLabel.pkl'
    with open(pkl_leYearsSinceLastPromotion, 'wb') as file:
        pickle.dump(leYearsSinceLastPromotion, file) 
    
    
    leYearsWithCurrManager = LabelEncoder()
    rawdf["YearsWithCurrManagerLabel"] = leYearsWithCurrManager.fit_transform(rawdf["YearsWithCurrManagerLabel"])
    pkl_leYearsWithCurrManager = './Model/pkl_leYearsWithCurrManagerLabel.pkl'
    with open(pkl_leYearsWithCurrManager, 'wb') as file:
        pickle.dump(leYearsWithCurrManager, file) 
        
    leAttrition = LabelEncoder()
    rawdf["Attrition"] =leAttrition.fit_transform(rawdf["Attrition"])    
    pkl_leAttrition = './Model/pkl_leAttrition.pkl'
    with open(pkl_leAttrition, 'wb') as file:
        pickle.dump(leAttrition, file) 
    return rawdf

LABEL_COLS = ["AgeLabel", "BusinessTravel", "DailyRatesLabel", "Department","DistanceFromHomeLabel",
               "EducationField", "Gender", "HourlyRateLabel",
               "JobRole", "MaritalStatus", 
               "MonthlyIncomeLabel", "MonthlyRateLabel", "NumCompaniesWorkedLabel", "OverTime", "PercentSalaryHikeLabel",
               "TotalWorkingYearsLabel","TrainingTimesLastYearLabel", "YearsAtCompanyLabel", "YearsInCurrentRoleLabel",
               "YearsSinceLastPromotionLabel", "YearsWithCurrManagerLabel"]

def TransformPredictCols(rawdf):

    for col in LABEL_COLS:
        pkl_file = './Model/pkl_le'+col+'.pkl'
        with open(pkl_file, 'rb') as file:
            rawdf[col] = pickle.load(file).transform(rawdf[col])
    return rawdf
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    