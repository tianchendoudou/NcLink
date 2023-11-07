import datetime
import pickle, methods, random, math,random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
import average
import gc
from pandas.core.frame import DataFrame
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE
from sklearn.ensemble import GradientBoostingClassifier
#from GCForest import *
from imblearn.metrics import geometric_mean_score
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # data = pd.read_csv("german_credit(test).csv")
    #
    # featurelist = ['account_check_status', 'credit_history', 'purpose', 'savings', 'present_emp_since',
    #                    'personal_status_sex', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job',
    #                    'telephone', 'foreign_worker']
    # for i in range(len(featurelist)):
    #     LE = LabelEncoder()
    #     LE.fit(data[featurelist[i]])
    #     newdata = LE.transform(data[featurelist[i]])
    #     data[featurelist[i]] = newdata
    # # XX = np.array(data.iloc[:, :-1])
    # X = np.array(data.loc[:,['account_check_status', 'credit_amount', 'duration_in_month', 'age', 'purpose','credit_history', 'property', 'present_emp_since', 'installment_as_income_perc','credits_this_bank']])
    # Y = data.iloc[:, -1].map(lambda x: 1 if x == 0 else -1)

    # data = pd.read_csv("new_finall.csv")
    # featurelist = ['f1', 'f2', 'f3', 'f4', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',
    #                'f15', 'f16', 'f17', 'f18', 'f19', 'f161', 'f162', 'f163', 'f164', 'f165', 'f166', 'f167', 'f168', 'f169',
    #                'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183',
    #                'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191', 'f192']
    # X = np.array(data.loc[:,featurelist])
    # Y = data.iloc[:, -1].map(lambda x: 1 if x == 0 else -1)
    # new_data = data.drop('state',axis=1)
    # X = np.array(new_data)

    # data = pd.read_csv("2018Q2.csv")
    # featurelist = data.columns.values.tolist()
    # loan_1 = data.copy()
    # data_1 = loan_1[loan_1['loan_status'] == 1].sample(200)
    # data_0 = loan_1[loan_1['loan_status'] == 0].sample(700)
    # data = pd.concat([data_1,data_0])
    # data = data.reindex(np.random.permutation(data.index))
    # data.reset_index(drop=True)
    # X = np.array(data.loc[:,featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    #glass数据
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\glass\glass.data', header=None)
    # data[10] = data[10].replace([1, 2, 3, 4, 5, 6, 7, 8, 9], ['0', '0', '0', '0', '1', '1', '1', '0', '0'])
    # data = data[data[10].isin(['0', '1'])]
    # data[10] = data[10].astype('int')
    # data = data.drop(0, axis=1)
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:,featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)


    #wine数据集
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\wine\wine.data', header=None)
    # featurelist = data.columns.values.tolist()[:-1]
    # data['label'] = data[0]
    # data = data.drop(0, axis=1)
    # data['label'] = data['label'].replace([1, 2, 3], ['0', '0', '1'])
    # data = data[data['label'].isin(['0', '1'])]
    # data['label'] = data['label'].astype('int')
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:,featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    # abalone数据集
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\\abalone\\abalone.data', header=None)
    # featurelist = data.columns.values.tolist()[:-1]
    # a = data[data[8] == 9]
    # b = data[data[8] == 18]
    # loaddata = pd.concat([a, b])
    # LE = LabelEncoder()
    # LE.fit(data[0])
    # newdata = LE.transform(loaddata[0])
    # loaddata[0] = newdata
    # loaddata[8] = loaddata[8].replace([18, 9],
    #                                   ['1', '0'])
    # loaddata = loaddata[loaddata[8].isin(['0', '1'])]
    # loaddata[8] = loaddata[8].astype('int')
    # featurelist = loaddata.columns.values.tolist()[:-1]
    # X = np.array(loaddata.loc[:, featurelist])
    # Y = loaddata.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    # # segment数据集
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\segment\segment.data', header=None)
    # featurelist = data.columns.values.tolist()[:-1]
    # data['label'] = data[0]
    # data = data.drop(0, axis=1)
    # data['label'] = data['label'].replace(['BRICKFACE', 'CEMENT', 'SKY', 'FOLIAGE', 'PATH', 'WINDOW', 'GRASS'],
    #                                       ['0', '0', '0', '0', '0', '0', '1'])
    # data = data[data['label'].isin(['0', '1'])]
    # data['label'] = data['label'].astype('int')
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:, featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    # yeast数据集
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\yeast\yeastdata.csv', header=None)
    # data = data.drop(0, axis=1)
    # featurelist = data.columns.values.tolist()[:-1]
    # data[9] = data[9].replace(['ME3', 'ME2', 'EXC', 'VAC', 'POX', 'ERL', 'ME1', 'CYT', 'NUC', 'MIT'],
    #                           ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0'])
    # data = data[data[9].isin(['0', '1'])]
    # data[9] = data[9].astype('int')
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:, featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    # breasttissue数据集
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\\breasttissue\\breasttissue.csv')
    # data = data.drop(['Case #'], axis=1)
    # data['label'] = data['Class']
    # data = data.drop(['Class'], axis=1)
    # data['label'] = data['label'].replace(['car', 'fad', 'adi', 'mas', 'gla', 'con'], ['1', '1', '0', '0', '0', '0'])
    # data = data[data['label'].isin(['0', '1'])]
    # data['label'] = data['label'].astype('int')
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:, featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    # pageblock数据
    # data = pd.read_csv('C:\\Users\hasee\Desktop\毕业论文\data\pageblock\pageblock.csv', header=None)
    # data[10] = data[10].replace([1, 2, 3, 4, 5], ['0', '0', '1', '1', '1'])
    # data = data[data[10].isin(['0', '1'])]
    # data[10] = data[10].astype('int')
    # featurelist = data.columns.values.tolist()[:-1]
    # X = np.array(data.loc[:, featurelist])
    # Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)

    #赛题数据
    path = 'E:\JupyterWorkplace\data\cs-training.csv'
    df = pd.read_csv(path)
    states = {'Unnamed: 0': 'ID',
              'SeriousDlqin2yrs': '好坏客户',
              'RevolvingUtilizationOfUnsecuredLines': '可用额度比值',
              'age': '年龄',
              'NumberOfTime30-59DaysPastDueNotWorse': '逾期30-59天笔数',
              'DebtRatio': '负债率',
              'MonthlyIncome': '月收入',
              'NumberOfOpenCreditLinesAndLoans': '信贷数量',
              'NumberOfTimes90DaysLate': '逾期90天笔数',
              'NumberRealEstateLoansOrLines': '固定资产贷款量',
              'NumberOfTime60-89DaysPastDueNotWorse': '逾期60-89天笔数',
              'NumberOfDependents': '家属数量'}
    df.rename(columns=states, inplace=True)
    df = df.set_index("ID", drop=True)  # 设置id列为索引列
    df = df.fillna({'月收入': df['月收入'].median()})
    df1 = df.dropna()
    df1 = df1[df1['可用额度比值'] < 1]
    df1 = df1[df1['年龄'] > 0]
    df1 = df1[df1['逾期30-59天笔数'] < 80]
    df1 = df1[df1['逾期60-89天笔数'] < 80]
    df1 = df1[df1['逾期90天笔数'] < 80]
    df1 = df1[df1['固定资产贷款量'] < 50]
    cut1 = pd.qcut(df1["可用额度比值"], 4, labels=False)
    cut2 = pd.qcut(df1["年龄"], 8, labels=False)
    bins3 = [-1, 0, 1, 3, 5, 13]
    cut3 = pd.cut(df1["逾期30-59天笔数"], bins3, labels=False)
    cut4 = pd.qcut(df1["负债率"], 3, labels=False)
    cut5 = pd.qcut(df1["月收入"], 4, labels=False)
    cut6 = pd.qcut(df1["信贷数量"], 4, labels=False)
    bins7 = [-1, 0, 1, 3, 5, 20]
    cut7 = pd.cut(df1["逾期90天笔数"], bins7, labels=False)
    bins8 = [-1, 0, 1, 2, 3, 33]
    cut8 = pd.cut(df1["固定资产贷款量"], bins8, labels=False)
    bins9 = [-1, 0, 1, 3, 12]
    cut9 = pd.cut(df1["逾期60-89天笔数"], bins9, labels=False)
    bins10 = [-1, 0, 1, 2, 3, 5, 21]
    cut10 = pd.cut(df1["家属数量"], bins10, labels=False)
    rate = df1["好坏客户"].sum() / (df1["好坏客户"].count() - df1["好坏客户"].sum())

    def get_woe_data(cut):
        grouped = df1["好坏客户"].groupby(cut, as_index=True).value_counts()
        woe = np.log(grouped.unstack().iloc[:, 1] / grouped.unstack().iloc[:, 0] / rate)
        return woe

    cut1_woe = get_woe_data(cut1)
    cut2_woe = get_woe_data(cut2)
    cut3_woe = get_woe_data(cut3)
    cut4_woe = get_woe_data(cut4)
    cut5_woe = get_woe_data(cut5)
    cut6_woe = get_woe_data(cut6)
    cut7_woe = get_woe_data(cut7)
    cut8_woe = get_woe_data(cut8)
    cut9_woe = get_woe_data(cut9)
    cut10_woe = get_woe_data(cut10)

    def get_IV_data(cut, cut_woe):
        grouped = df1["好坏客户"].groupby(cut, as_index=True).value_counts()
        cut_IV = ((grouped.unstack().iloc[:, 1] / df1["好坏客户"].sum() - grouped.unstack().iloc[:, 0] / (
                    df1["好坏客户"].count() - df1["好坏客户"].sum())) * cut_woe).sum()
        return cut_IV

    # 计算各分组的IV值
    cut1_IV = get_IV_data(cut1, cut1_woe)
    cut2_IV = get_IV_data(cut2, cut2_woe)
    cut3_IV = get_IV_data(cut3, cut3_woe)
    cut4_IV = get_IV_data(cut4, cut4_woe)
    cut5_IV = get_IV_data(cut5, cut5_woe)
    cut6_IV = get_IV_data(cut6, cut6_woe)
    cut7_IV = get_IV_data(cut7, cut7_woe)
    cut8_IV = get_IV_data(cut8, cut8_woe)
    cut9_IV = get_IV_data(cut9, cut9_woe)
    cut10_IV = get_IV_data(cut10, cut10_woe)

    df_new = pd.DataFrame()  # 新建df_new存放woe转换后的数据
    def replace_data(cut, cut_woe):
        a = []
        for i in cut.unique():
            a.append(i)
            a.sort()
        for m in range(len(a)):
            cut.replace(a[m], cut_woe.values[m], inplace=True)
        return cut

    df_new["好坏客户"] = df1["好坏客户"]
    df_new["可用额度比值"] = replace_data(cut1, cut1_woe)
    df_new["年龄"] = replace_data(cut2, cut2_woe)
    df_new["逾期30-59天笔数"] = replace_data(cut3, cut3_woe)
    df_new["负债率"] = replace_data(cut4, cut4_woe)
    df_new["月收入"] = replace_data(cut5, cut5_woe)
    df_new["信贷数量"] = replace_data(cut6, cut6_woe)
    df_new["逾期90天笔数"] = replace_data(cut7, cut7_woe)
    df_new["固定资产贷款量"] = replace_data(cut8, cut8_woe)
    df_new["逾期60-89天笔数"] = replace_data(cut9, cut9_woe)
    df_new["家属数量"] = replace_data(cut10, cut10_woe)
    df_new["label"] = df_new["好坏客户"]

    featurelist = ['可用额度比值', '年龄', '逾期30-59天笔数', '负债率', '月收入', '信贷数量', '逾期90天笔数', '固定资产贷款量',
                   '逾期60-89天笔数', '家属数量']
    chargeoff = df_new[df_new.label == 0]
    fullypaid = df_new[df_new.label == 1]
    data = chargeoff.append(fullypaid)
    a = data.loc[:,featurelist]
    X = np.array(data.loc[:,featurelist])
    Y = data.iloc[:, -1].map(lambda x: -1 if x == 0 else 1)



    # sm = SMOTE(random_state=3)
    # ada = ADASYN(random_state=3)
    # bsm = BorderlineSMOTE(random_state=3)
    # X_resampled_smote, y_resampled_smote = sm.fit_sample(X, Y)
    # X_resampled_adasyn, y_resampled_adasyn = ada.fit_sample(X, Y)
    # X_resampled_bsmote, y_resampled_bsmote = bsm.fit_sample(X, Y)


    inX = list(X)
    inY = list(Y)
    X_g, Y_g = methods.MWMOTE(inX, inY, 5000)
    z = np.array(X_g)
    w = pd.Series(Y_g)

    YY = list(Y)
    YY.extend(Y_g)
    fin_y = pd.Series(YY)
    fin_X = np.vstack((X,z))


RF_test_auc_list = []
SVM_test_auc_list = []
SMOTE_RF_test_auc_list = []
BSMOTE_RF_test_auc_list = []
ADA_RF_test_auc_list = []
SMOTE_SVM_test_auc_list = []
BSMOTE_SVM_test_auc_list = []
ADA_SVM_test_auc_list = []
SMOTE_GBDT_test_auc_list = []
BSMOTE_GBDT_test_auc_list = []
ADA_GBDT_test_auc_list = []
MRF_test_auc_list = []
GBDT_test_auc_list = []
MGBDT_test_auc_list = []
MSVM_test_auc_list = []

RF_test_f1_list = []
SVM_test_f1_list = []
SMOTE_RF_test_f1_list = []
BSMOTE_RF_test_f1_list = []
ADA_RF_test_f1_list = []
SMOTE_SVM_test_f1_list = []
BSMOTE_SVM_test_f1_list = []
ADA_SVM_test_f1_list = []
MRF_test_f1_list = []
GBDT_test_f1_list = []
MGBDT_test_f1_list = []
MSVM_test_f1_list = []
SMOTE_GBDT_test_f1_list = []
BSMOTE_GBDT_test_f1_list = []
ADA_GBDT_test_f1_list = []

RF_test_recall_list = []
SVM_test_recall_list = []
GBDT_test_recall_list = []
MRF_test_recall_list = []
MGBDT_test_recall_list = []
MSVM_test_recall_list = []
SMOTE_RF_test_recall_list = []
BSMOTE_RF_test_recall_list = []
ADA_RF_test_recall_list = []
SMOTE_GBDT_test_recall_list = []
BSMOTE_GBDT_test_recall_list = []
ADA_GBDT_test_recall_list = []
SMOTE_SVM_test_recall_list = []
BSMOTE_SVM_test_recall_list = []
ADA_SVM_test_recall_list = []

SVM_test_acu_list = []
ks_list = []

from imblearn.metrics import geometric_mean_score

def compute_ks(label,predict_proba):
    fpr,tpr,thresholds = roc_curve(label,predict_proba)
    return max(tpr-fpr)

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.3)
    # X_train_bsmote, X_test_bsmote, y_train_bsmote, y_test_bsmote = train_test_split(X_resampled_bsmote, y_resampled_bsmote,test_size=0.3)
    # X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(X_resampled_adasyn, y_resampled_adasyn, test_size=0.3)
    nX_train, nX_test, ny_train, ny_test = train_test_split(fin_X, fin_y, test_size=0.3)


    RF = RandomForestClassifier(n_estimators = 5,max_features=5)
    MRF = RandomForestClassifier(n_estimators = 10)
    NB = GaussianNB()
    #SVM = gcForest(shape_1X = [1,12],window=[2])#SVC(kernel= 'rbf',probability=True)
    MSVM = SVC(kernel= 'rbf',probability=True)#KNeighborsClassifier(n_neighbors=5)
    KNN = SVC(probability=True)
    GBDT = GradientBoostingClassifier(n_estimators=5)
    MGBDT = GradientBoostingClassifier(n_estimators=5)


    #单纯的随机森林
    # RF.fit(X_train, y_train)
    # RF_test_prob = RF.predict_proba(X_test)[:,1]
    # RF_test_pre = RF.predict(X_test)
    # RFauc = metrics.roc_auc_score(list(y_test), RF_test_prob)
    # RF_test_auc_list.append(RFauc)
    # RF_test_f1_list.append(metrics.f1_score(list(y_test), RF_test_pre))
    # RF_test_recall_list.append(metrics.recall_score(list(y_test), RF_test_pre))
    #单纯的SVM
    # SVM.fit(X_train, y_train)
    # # #SVM_test_prob = SVM.predict_proba(X_test)[:,1]
    # SVM_test_pre = SVM.predict(X_test)
    # # #SVMauc = metrics.roc_auc_score(y_test, SVM_test_prob)
    # # #SVM_test_auc_list.append(SVMauc)
    # SVM_test_f1_list.append(metrics.f1_score(list(y_test), SVM_test_pre))
    # SVM_test_recall_list.append(metrics.recall_score(list(y_test), SVM_test_pre))
    # SVM_test_acu_list.append(metrics.accuracy_score(list(y_test), SVM_test_pre))
    #单纯的GBDT
    # GBDT.fit(X_train, y_train)
    # GBDT_test_prob = GBDT.predict_proba(X_test)[:,1]
    # GBDT_test_pre = GBDT.predict(X_test)
    # GBDTauc = metrics.roc_auc_score(list(y_test), GBDT_test_prob)
    # GBDT_test_auc_list.append(GBDTauc)
    # GBDT_test_f1_list.append(metrics.f1_score(list(y_test), GBDT_test_pre))
    # GBDT_test_recall_list.append(metrics.recall_score(list(y_test),  GBDT_test_pre))
    # #SMOTE处理后的随机森林
    # RF.fit(X_train_smote, y_train_smote)
    # SMOTE_RF_test_prob = RF.predict_proba(X_test)[:,1]
    # SMOTE_RFauc = metrics.roc_auc_score(list(y_test), SMOTE_RF_test_prob)
    # SMOTE_RF_test_auc_list.append(SMOTE_RFauc)
    # SMOTE_RF_test_pre = RF.predict(X_test)
    # SMOTE_RF_test_f1_list.append(metrics.f1_score(list(y_test), SMOTE_RF_test_pre))
    # SMOTE_RF_test_recall_list.append(metrics.recall_score(list(y_test),  SMOTE_RF_test_pre))
    # #BSMOTE处理后的随机森林
    # RF.fit(X_train_bsmote, y_train_bsmote)
    # BSMOTE_RF_test_prob = RF.predict_proba(X_test)[:,1]
    # BSMOTE_RFauc = metrics.roc_auc_score(list(y_test), BSMOTE_RF_test_prob)
    # BSMOTE_RF_test_auc_list.append(BSMOTE_RFauc)
    # BSMOTE_RF_test_pre = RF.predict(X_test)
    # BSMOTE_RF_test_f1_list.append(metrics.f1_score(list(y_test), BSMOTE_RF_test_pre))
    # BSMOTE_RF_test_recall_list.append(metrics.recall_score(list(y_test),  BSMOTE_RF_test_pre))
    # #ADA处理后的随机森林
    # RF.fit(X_train_ada, y_train_ada)
    # ADA_RF_test_prob = RF.predict_proba(X_test)[:,1]
    # ADA_RFauc = metrics.roc_auc_score(list(y_test), ADA_RF_test_prob)
    # ADA_RF_test_auc_list.append(ADA_RFauc)
    # ADA_RF_test_pre = RF.predict(X_test)
    # ADA_RF_test_f1_list.append(metrics.f1_score(list(y_test), ADA_RF_test_pre))
    # ADA_RF_test_recall_list.append(metrics.recall_score(list(y_test),  ADA_RF_test_pre))
    # #经SMOTE处理后的GBDT
    # GBDT.fit(X_train_smote, y_train_smote)
    # SMOTE_GBDT_test_prob = GBDT.predict_proba(X_test)[:,1]
    # SMOTE_GBDTauc = metrics.roc_auc_score(list(y_test), SMOTE_GBDT_test_prob)
    # SMOTE_GBDT_test_auc_list.append(SMOTE_GBDTauc)
    # SMOTE_GBDT_test_pre = GBDT.predict(X_test)
    # SMOTE_GBDT_test_f1_list.append(metrics.f1_score(list(y_test), SMOTE_GBDT_test_pre))
    # SMOTE_GBDT_test_recall_list.append(metrics.recall_score(list(y_test),  SMOTE_GBDT_test_pre))
    #
    # GBDT.fit(X_train_bsmote, y_train_bsmote)
    # BSMOTE_GBDT_test_prob = GBDT.predict_proba(X_test)[:,1]
    # BSMOTE_GBDTauc = metrics.roc_auc_score(list(y_test), BSMOTE_GBDT_test_prob)
    # BSMOTE_GBDT_test_auc_list.append(BSMOTE_GBDTauc)
    # BSMOTE_GBDT_test_pre = GBDT.predict(X_test)
    # BSMOTE_GBDT_test_f1_list.append(metrics.f1_score(list(y_test), BSMOTE_GBDT_test_pre))
    # BSMOTE_GBDT_test_recall_list.append(metrics.recall_score(list(y_test),  BSMOTE_GBDT_test_pre))
    #
    # GBDT.fit(X_train_ada, y_train_ada)
    # ADA_GBDT_test_prob = GBDT.predict_proba(X_test)[:,1]
    # ADA_GBDTauc = metrics.roc_auc_score(list(y_test), ADA_GBDT_test_prob)
    # ADA_GBDT_test_auc_list.append(ADA_GBDTauc)
    # ADA_GBDT_test_pre = GBDT.predict(X_test)
    # ADA_GBDT_test_f1_list.append(metrics.f1_score(list(y_test), ADA_GBDT_test_pre))
    # ADA_GBDT_test_recall_list.append(metrics.recall_score(list(y_test),  ADA_GBDT_test_pre))

    #SMOTE处理后的随机森林
    # SVM.fit(X_train_smote, y_train_smote)
    # SMOTE_SVM_test_prob = SVM.predict_proba(X_test)[:,1]
    # SMOTE_SVMauc = metrics.roc_auc_score(y_test, SMOTE_SVM_test_prob)
    # SMOTE_SVM_test_auc_list.append(SMOTE_SVMauc)
    # SMOTE_SVM_test_pre = SVM.predict(X_test)
    # SMOTE_SVM_test_f1_list.append(metrics.f1_score(list(y_test), SMOTE_SVM_test_pre))
    # SMOTE_SVM_test_recall_list.append(metrics.recall_score(list(y_test),  SMOTE_SVM_test_pre))
    # #BSMOTE处理后的随机森林
    # SVM.fit(X_train_bsmote, y_train_bsmote)
    # BSMOTE_SVM_test_prob = SVM.predict_proba(X_test)[:,1]
    # BSMOTE_SVMauc = metrics.roc_auc_score(y_test, BSMOTE_SVM_test_prob)
    # BSMOTE_SVM_test_auc_list.append(BSMOTE_SVMauc)
    # BSMOTE_SVM_test_pre = SVM.predict(X_test)
    # BSMOTE_SVM_test_f1_list.append(metrics.f1_score(list(y_test), BSMOTE_SVM_test_pre))
    # BSMOTE_SVM_test_recall_list.append(metrics.recall_score(list(y_test),  BSMOTE_SVM_test_pre))
    # #ADA处理后的随机森林
    # SVM.fit(X_train_ada, y_train_ada)
    # ADA_SVM_test_prob = SVM.predict_proba(X_test)[:,1]
    # ADA_SVMauc = metrics.roc_auc_score(y_test, ADA_SVM_test_prob)
    # ADA_SVM_test_auc_list.append(ADA_SVMauc)
    # ADA_SVM_test_pre = SVM.predict(X_test)
    # ADA_SVM_test_f1_list.append(metrics.f1_score(list(y_test), ADA_SVM_test_pre))
    # ADA_SVM_test_recall_list.append(metrics.recall_score(list(y_test),  ADA_SVM_test_pre))



    MRF.fit(nX_train, ny_train)
    MRF_test_prob = MRF.predict_proba(X_test)[:,1]
    MRF_test_pre = MRF.predict(X_test)
    MRFauc = metrics.roc_auc_score(list(y_test), MRF_test_prob)
    MRF_test_auc_list.append(MRFauc)
    MRF_test_f1_list.append(metrics.f1_score(list(y_test), MRF_test_pre))
    #MRF_test_recall_list.append(metrics.recall_score(list(y_test), MRF_test_pre))
    MRF_test_recall_list.append(geometric_mean_score(list(y_test), MRF_test_pre,average='weighted'))
    ks_list.append(compute_ks(list(y_test), MRF_test_prob))

    # MGBDT.fit(nX_train, ny_train)
    # MGBDT_test_prob = MGBDT.predict_proba(X_test)[:,1]
    # MGBDT_test_pre = MGBDT.predict(X_test)
    # MGBDTauc = metrics.roc_auc_score(list(y_test), MGBDT_test_prob)
    # MGBDT_test_auc_list.append(MGBDTauc)
    # MGBDT_test_f1_list.append(metrics.fowlkes_mallows_score(list(y_test), MGBDT_test_pre))
    # MGBDT_test_recall_list.append(metrics.recall_score(list(y_test), MGBDT_test_pre))

    # MSVM.fit(nX_train, ny_train)
    # MSVM_test_prob = MSVM.predict_proba(X_test)[:,1]
    # MSVM_test_pre = MSVM.predict(X_test)
    # MSVMauc = metrics.roc_auc_score(list(y_test), MSVM_test_prob)
    # MSVM_test_auc_list.append(MSVMauc)
    # MSVM_test_f1_list.append(metrics.f1_score(list(y_test), MSVM_test_pre))
    # MSVM_test_recall_list.append(metrics.recall_score(list(y_test), MSVM_test_pre))



# print ('MRF: max ACU with %f' % max(MRF_test_auc_list))
# print ('MRF: min ACU with %f' % min(MRF_test_auc_list))
print ('MRF: aver ACU with %f' % average.averagenum(MRF_test_auc_list))
print ('MRF: aver f1 with %f' % average.averagenum(MRF_test_f1_list))
print ('MRF: aver gmean with %f' % average.averagenum(MRF_test_recall_list))
print ('MRF: aver ks with %f' % average.averagenum(ks_list))

# print ('MSVM: max ACU with %f' % max(MSVM_test_auc_list))
# print ('MSVM: min ACU with %f' % min(MSVM_test_auc_list))
# print ('MSVM: aver ACU with %f' % average.averagenum(MSVM_test_auc_list))
# print ('MSVM: aver f1 with %f' % average.averagenum(MSVM_test_f1_list))
# print ('MSVM: aver recall with %f' % average.averagenum(MSVM_test_recall_list))

# print ('MGBDT: max ACU with %f' % max(MGBDT_test_auc_list))
# print ('MGBDT: min ACU with %f' % min(MGBDT_test_auc_list))
# print ('MGBDT: aver ACU with %f' % average.averagenum(MGBDT_test_auc_list))
# print ('MGBDT: aver f1 with %f' % average.averagenum(MGBDT_test_f1_list))
# print ('MGBDT: aver recall with %f' % average.averagenum(MGBDT_test_recall_list))
#
# # print('RF: max ACU with %f' % max(RF_test_auc_list))
# # print ('RF: min ACU with %f' % min(RF_test_auc_list))
# print ('RF: aver ACU with %f' % average.averagenum(RF_test_auc_list))
# print ('RF: aver f1 with %f' % average.averagenum(RF_test_f1_list))
# print ('RF: aver recall with %f' % average.averagenum(RF_test_recall_list))
#
# # print('SVM: max ACU with %f' % max(SVM_test_auc_list))
# # print ('SVM: min ACU with %f' % min(SVM_test_auc_list))
# # print ('SVM: aver ACU with %f' % average.averagenum(SVM_test_auc_list))
# # print ('SVM: aver f1 with %f' % average.averagenum(SVM_test_f1_list))
# # print ('SVM: aver recall with %f' % average.averagenum(SVM_test_recall_list))
# # print ('SVM: aver acu with %f' % average.averagenum(SVM_test_acu_list))
#
# # print('GBDT: max ACU with %f' % max(GBDT_test_auc_list))
# # print ('GBDT: min ACU with %f' % min(GBDT_test_auc_list))
# print ('GBDT: aver ACU with %f' % average.averagenum(GBDT_test_auc_list))
# print ('GBDT: aver f1 with %f' % average.averagenum(GBDT_test_f1_list))
# print ('GBDT: aver recall with %f' % average.averagenum(GBDT_test_recall_list))
#
# # print('SMOTE_GBDT: max ACU with %f' % max(SMOTE_GBDT_test_auc_list))
# # print ('SMOTE_GBDT: min ACU with %f' % min(SMOTE_GBDT_test_auc_list))
# print ('SMOTE_GBDT: aver ACU with %f' % average.averagenum(SMOTE_GBDT_test_auc_list))
# print ('SMOTE_GBDT: aver f1 with %f' % average.averagenum(SMOTE_GBDT_test_f1_list))
# print ('SMOTE_GBDT: aver recall with %f' % average.averagenum(SMOTE_GBDT_test_recall_list))
#
# # print('BSMOTE_GBDT: max ACU with %f' % max(BSMOTE_GBDT_test_auc_list))
# # print ('BSMOTE_GBDT: min ACU with %f' % min(BSMOTE_GBDT_test_auc_list))
# print ('BSMOTE_GBDT: aver ACU with %f' % average.averagenum(BSMOTE_GBDT_test_auc_list))
# print ('BSMOTE_GBDT: aver f1 with %f' % average.averagenum(BSMOTE_GBDT_test_f1_list))
# print ('BSMOTE_GBDT: aver recall with %f' % average.averagenum(BSMOTE_GBDT_test_recall_list))
#
# # print('ADA_GBDT: max ACU with %f' % max(ADA_GBDT_test_auc_list))
# # print ('ADA_GBDT: min ACU with %f' % min(ADA_GBDT_test_auc_list))
# print ('ADA_GBDT: aver ACU with %f' % average.averagenum(ADA_GBDT_test_auc_list))
# print ('ADA_GBDT: aver f1 with %f' % average.averagenum(ADA_GBDT_test_f1_list))
# print ('ADA_GBDT: aver recall with %f' % average.averagenum(ADA_GBDT_test_recall_list))
#
# # print('SMOTE_RF: max ACU with %f' % max(SMOTE_RF_test_auc_list))
# # print ('SMOTE_RF: min ACU with %f' % min(SMOTE_RF_test_auc_list))
# print ('SMOTE_RF: aver ACU with %f' % average.averagenum(SMOTE_RF_test_auc_list))
# print ('SMOTE_RF: aver f1 with %f' % average.averagenum(SMOTE_RF_test_f1_list))
# print ('SMOTE_RF: aver recall with %f' % average.averagenum(SMOTE_RF_test_recall_list))
#
# # print('BSMOTE_RF: max ACU with %f' % max(BSMOTE_RF_test_auc_list))
# # print ('BSMOTE_RF: min ACU with %f' % min(BSMOTE_RF_test_auc_list))
# print ('BSMOTE_RF: aver ACU with %f' % average.averagenum(BSMOTE_RF_test_auc_list))
# print ('BSMOTE_RF: aver f1 with %f' % average.averagenum(BSMOTE_RF_test_f1_list))
# print ('BSMOTE_RF: aver recall with %f' % average.averagenum(BSMOTE_RF_test_recall_list))
#
# # print('ADA_RF: max ACU with %f' % max(ADA_RF_test_auc_list))
# # print ('ADA_RF: min ACU with %f' % min(ADA_RF_test_auc_list))
# print ('ADA_RF: aver ACU with %f' % average.averagenum(ADA_RF_test_auc_list))
# print ('ADA_RF: aver f1 with %f' % average.averagenum(ADA_RF_test_f1_list))
# print ('ADA_RF: aver recall with %f' % average.averagenum(ADA_RF_test_recall_list))

# print('SMOTE_SVM: max ACU with %f' % max(SMOTE_SVM_test_auc_list))
# print ('SMOTE_SVM: min ACU with %f' % min(SMOTE_SVM_test_auc_list))
# print ('SMOTE_SVM: aver ACU with %f' % average.averagenum(SMOTE_SVM_test_auc_list))
# print ('SMOTE_SVM: aver f1 with %f' % average.averagenum(SMOTE_SVM_test_f1_list))
# print ('SMOTE_SVM: aver recall with %f' % average.averagenum(SMOTE_SVM_test_recall_list))
#
# print('BSMOTE_SVM: max ACU with %f' % max(BSMOTE_SVM_test_auc_list))
# print ('BSMOTE_SVM: min ACU with %f' % min(BSMOTE_SVM_test_auc_list))
# print ('BSMOTE_SVM: aver ACU with %f' % average.averagenum(BSMOTE_SVM_test_auc_list))
# print ('BSMOTE_SVM: aver f1 with %f' % average.averagenum(BSMOTE_SVM_test_f1_list))
# print ('BSMOTE_SVM: aver recall with %f' % average.averagenum(BSMOTE_SVM_test_recall_list))
#
# print('ADA_SVM: max ACU with %f' % max(ADA_SVM_test_auc_list))
# print ('ADA_SVM: min ACU with %f' % min(ADA_SVM_test_auc_list))
# print ('ADA_SVM: aver ACU with %f' % average.averagenum(ADA_SVM_test_auc_list))
# print ('ADA_SVM: aver f1 with %f' % average.averagenum(ADA_SVM_test_f1_list))
# print ('ADA_SVM: aver recall with %f' % average.averagenum(ADA_SVM_test_recall_list))

# print('MSVM: max ACU with %f' % max(SVM_test_auc_list))
# print ('MSVM: min ACU with %f' % min(SVM_test_auc_list))
# print ('MSVM: aver ACU with %f' % average.averagenum(SVM_test_auc_list))


endtime = datetime.datetime.now()
print (endtime - starttime).seconds



#绘制ROC图像
# fpr_rf, tpr_rf, _ = roc_curve(y_test, RF_test_prob)
# fpr_nf, tpr_nf, _ = roc_curve(y_test, NB_test_prob)
# fpr_mrf, tpr_mrf, _ = roc_curve(y_test, MRF_test_prob)
# fpr_srf, tpr_srf, _ = roc_curve(y_test, SMOTE_RF_test_prob)
# fpr_bsrf, tpr_bsrf, _ = roc_curve(y_test, BSMOTE_RF_test_prob)
# fpr_adrf, tpr_adrf, _ = roc_curve(y_test, ADA_RF_test_prob)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rf, tpr_rf, label='RF',linestyle='--', marker='*',color = 'black' )
# # plt.plot(fpr_srf, tpr_srf, label='SMOTE+RF',linestyle='--', marker='*',color = 'blue' )
# # plt.plot(fpr_bsrf, tpr_bsrf, label='BordlineSMOTE+RF',linestyle='--', marker='+',color = 'red' )
# # plt.plot(fpr_adrf, tpr_adrf, label='ADASYN+RF',linestyle='--', marker='^',color = 'yellow' )
# plt.plot(fpr_nf, tpr_nf, label='NB',linestyle='--', marker='+',color = 'black' )
# plt.plot(fpr_mrf, tpr_mrf, label='MWMOTE + RF',color = 'black')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig('图2.svg', dpi=600)
# plt.show()



