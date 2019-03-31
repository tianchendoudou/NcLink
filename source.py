
import pickle,random, math,random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def source():
    data = pd.read_csv("german_credit(test).csv")
    featurelist = ['account_check_status', 'credit_history', 'purpose', 'savings', 'present_emp_since',
               'personal_status_sex', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job',
               'telephone', 'foreign_worker']
    for i in range(len(featurelist)):
        LE = LabelEncoder()
        LE.fit(data[featurelist[i]])
        newdata = LE.transform(data[featurelist[i]])
        data[featurelist[i]] = newdata
# XX = np.array(data.iloc[:, :-1])
    X = np.array(data.loc[:
             ,['account_check_status', 'credit_amount', 'duration_in_month', 'age', 'purpose' ,'credit_history', 'property', 'present_emp_since', 'installment_as_income_perc' ,'credits_this_bank']])
    Y = data.iloc[:, -1].map(lambda x: 1 if x == 0 else -1)

    inX = list(X)
    inY = list(Y)

    S_min, S_maj = [], []
    for index, i in enumerate(inY):
        if i < 0:
            S_maj.append(index)
        else:
            S_min.append(index)
    return inX,S_min