import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib
import json

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns=None
pd.options.display.float_format='{:.2f}'.format
#os.listdir('Salary_Data.csv')
data = pd.read_csv(r'C:\Users\kalas\OneDrive\Desktop\Scorekart\Salary_Data.csv')
df=data.copy()
#df.info()
df.columns.str.strip()
df.isna().sum()
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum().sum()
df.duplicated().sum()
df=df.drop_duplicates()
df.describe().T
df.describe(include='object').T
df.hist(bins=20, color='red', figsize=(20,10));
df['Age']=pd.to_numeric(df['Age'], errors='coerce')
df['Age_Categ']=pd.cut(df['Age'], bins=[19,30,40, 50,60, np.inf], labels=['Twenties','Thirties','Forties','Fifties','Above_Sixty'])
df['Years_Exper']=pd.cut(df['Years of Experience'], bins=[-1,5,10,15,20, np.inf], labels=['0-5','6-10','11-15','16-20','above 20'])
df['Salary_Categ'] = pd.cut(df['Salary'], bins=[0, 50000, 100000, 150000, 200000, np.inf], labels=['Low', 'Medium', 'High', 'Very High', 'Top Tier'])
df
obj=df[['Gender','Age_Categ','Education Level','Job Title','Years_Exper','Salary_Categ']]
for col in df.select_dtypes("object"):
    print(f'\n----------{col}--------------')
    print(f'\n{df[col].value_counts().reset_index().sort_values(by="count", ascending=False)[:25]}')
    print(f'\n Number of Items by {col}= {df[col].nunique()}\n')
    print("::"*33)
df['Education Level']=df['Education Level'].replace('phD', 'PhD')
df['Education Level']=df['Education Level'].replace("Bachelor's Degree", "Bachelor's")
df['Education Level']=df['Education Level'].replace("Master's Degree", "Master's")
df['Education Level']=df['Education Level'].replace("Bachelor's", "Bachelor")
df['Education Level']=df['Education Level'].replace("Master's", "Master")
df['Education Level'].value_counts(normalize=True)
num=df.select_dtypes('number')
plt.figure(figsize=(20,8))
for ind, col in enumerate(num):
    plt.subplot(1,3,1+ind)
    plt.title("Box Plot of " + col, fontsize=20)
    sns.boxplot(data=df, y=col)
plt.tight_layout()
plt.show();
num = ['Age','Years of Experience','Salary']
for col in num:
    series = pd.to_numeric(df[col], errors='coerce')   # to confirm there is no str values
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, upp = q1 - 1.5*iqr, q3 + 1.5*iqr
    number_outliers = series[(series < low) | (series > upp)].shape[0]
    
    print(f"\nNumber of Outliers of {col} = {number_outliers}\n")
    # display highest and lowest values to confirm if this is a real outlier or not
    print("Highest Values")
    print(f'{df[col].value_counts().reset_index().sort_values(by=col, ascending=False)[:5]}\n') 
    print("Lowest Values")
    print(f'{df[col].value_counts().reset_index().sort_values(by=col, ascending=True)[:5]}\n')
    print(f'\nMax {col}= {df[col].max()}')
    print(f'Min {col}= {df[col].min()}\n')
    print("::"*33)
for col in num:
    Q1,Q3= df[col].quantile([0.25,0.75])
    IQR= q3-q1
    Low, Upp= Q1-1.5*IQR, Q3+1.5*IQR
    df=df[(df[col]>=Low) & (df[col]<=Upp)]
Female=df[df['Gender']=='Female']
Male=df[df['Gender']=='Male']
plt.figure(figsize=(25,10))
g= sns.catplot(kind='bar', data=Female, x='Years_Exper', y='Salary', hue='Education Level', col='Salary_Categ')
g.set_xticklabels(rotation=45,fontsize=15)
plt.suptitle("Female Variables Relationships", fontsize=25, color='red', weight='bold', y=1.10) 
plt.show();
plt.figure(figsize=(25,10))
g= sns.catplot(kind='bar', data=Male, x='Years_Exper', y='Salary', hue='Education Level', col='Salary_Categ')
g.set_xticklabels(rotation=45, fontsize=15)
plt.suptitle("Male Variables Relationships", fontsize=25, color='red', weight='bold', y=1.10) 
plt.show();
plt.figure(figsize=(20,8))
for ind, col in enumerate(num[:2]):
    top_num=df[col].value_counts().index[:10]
    ft=df[df[col].isin(top_num)]
    plt.subplot(1,2,1+ind)
    ax=sns.barplot(data=ft, x=col, y='Salary', hue='Gender', estimator=np.mean)
    plt.tight_layout()
plt.suptitle('Top Frequent Numeric Values vs Average Salary', y=1.03, weight='bold', fontsize=20, color='red')    
plt.show();
plt.figure(figsize=(20,8))
for ind, col in enumerate(num[:2]):
    top_num=df[col].value_counts().index[-10:]
    ft=df[df[col].isin(top_num)]
    plt.subplot(1,2,1+ind)
    sns.barplot(data=ft, x=col, y='Salary', hue='Gender', estimator=np.mean)
    plt.xticks(rotation=45, size=12)
    plt.tight_layout()
plt.suptitle('Least Frequent Numeric Values vs Average Salary', y=1.03, weight='bold', fontsize=20, color='red')    
plt.show();
plt.figure(figsize=(20,8))
sns.pairplot(df[['Age', 'Years of Experience', 'Salary']], kind='reg')
plt.tight_layout();
corr=df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
dm=df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary',]]
dm=pd.get_dummies(dm, columns=['Gender', 'Education Level'], drop_first=True, dtype=int)
la=LabelEncoder()
dm['Job_Title_encd']=la.fit_transform(dm['Job Title'])


job_mapping={'Job Title': la.classes_.tolist()}


with open('job_mapping.json', 'w') as f:
    json.dump(job_mapping, f, indent=4)
dm.head(4)
dm.columns
x = dm[['Age', 'Years of Experience', 'Gender_Male', 'Gender_Other', 'Education Level_High School', 'Education Level_Master',
       'Education Level_PhD', 'Job_Title_encd']]
y = dm['Salary']

model = sm.OLS(y, x).fit()  
model.summary()
jop_mapping=(dict(zip(range(len(la.classes_)), la.classes_)))
jop_mapping
model.model.exog_names
new_data=pd.DataFrame([{'Age': 39, 'Years of Experience': 5, 'Gender_Male': 1, 'Gender_Other':0,
       'Education Level_High School':0, 'Education Level_Master':1, 'Education Level_PhD':0, 'Job_Title_encd':20}]) 
prediction=model.predict(new_data)
print(f'Predicted Salery= ${prediction[0]:,.2f}')
y_true = dm['Salary']
y_pred = model.predict(dm[x.columns]) 

# compare first 10 columns between predicted salary and actual salary
print(pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}).head(10))

from sklearn.metrics import mean_absolute_error, mean_squared_error
# the measurement of accuracy 
print(f"\nMean Absolute Error: {mean_absolute_error(y_true, y_pred):.2f}")
print(f"Mean Square Error: {mean_squared_error(y_true, y_pred, squared=False):.2f}")
print(f"Mean Absolute Percantage Error: {np.mean(np.abs((y_true - y_pred) / y_true)) * 100:.2f} %\n")
X = dm[['Age', 'Years of Experience', 'Gender_Male', 'Gender_Other', 'Education Level_High School', 'Education Level_Master',
       'Education Level_PhD', 'Job_Title_encd']]
y = dm['Salary']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.33, random_state=42)
models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(),"Gradient Boosting":GradientBoostingRegressor(),"Decision Tree":DecisionTreeRegressor(),"KNN":KNeighborsRegressor()}
results={}
best_cv=-1
best_name=None
best_estimator=None
for name, model in models.items():
    model.fit(X_train, y_train)
    y_predict= model.predict(X_test)
    r2=r2_score(y_test, y_predict)
    mse=mean_squared_error(y_test, y_predict)
    cv_scores=cross_val_score(model, X,y, cv=5, scoring='r2')
    mean_cv=np.mean(cv_scores)
    results[name]=mean_cv
    print(f'\n-----------{name}------------------')
    print(f'R2: {r2:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'CVS: {cv_scores}')
    print(f'Mean CV: {mean_cv:.2f}')
    if mean_cv>best_cv:
        best_cv=mean_cv
        best_name=name
        best_estimator=model

final_model=best_estimator
print(f'\n Best Model is: {best_name}, CV: {best_cv:.2f}')

joblib.dump(final_model, 'salary_predict.pkl')
print(f'\n Best Model Saved as "salary.predict.pkl"')
sk_model=joblib.load('salary_predict.pkl')
sk_model
job_mapping=dict(zip(range(len(la.classes_)), la.classes_))
job_mapping
sk_new_data=pd.DataFrame([{'Age': 39, 'Years of Experience': 5, 'Gender_Male': 1, 'Gender_Other':0,
       'Education Level_High School':0, 'Education Level_Master':1, 'Education Level_PhD':0, 'Job_Title_encd':20}]) 
prediction=model.predict(sk_new_data)
print(f'Predicted Salery= $ {prediction[0]:,.2f}')

