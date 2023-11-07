import pandas as pd
import numpy as np
df=pd.DataFrame({'班级':['A','B','B']
               ,'姓名':['马二','马二','马二']
               ,'语文':[2,6,7]
               ,'数学':[1,7,8]}).round()

print(df)

a = df.groupby(['班级','姓名']).agg({'语文':['sum'],'数学':['mean']})
a.columns = ['_m2_'.join(col).strip() for col in a.columns.values]
a.reset_index(inplace=True)
print(a)


b = a.groupby('姓名').agg({'班级':['max']})
b.columns = ['_1_'.join(col).strip() for col in b.columns.values]
b.reset_index(inplace=True)
print(b)