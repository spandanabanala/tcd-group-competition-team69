import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from category_encoders import TargetEncoder
import lightgbm as lgb

target = 'Total Yearly Income [EUR]'
training_columns = [
    'Year of Record',
    'Age',
    'Gender',
    'Country',
    'Size of City',
    'Housing Situation',
    'University Degree',
    'Crime Level in the City of Employement',
    'Work Experience in Current Job [years]',
    'Satisfation with employer',
    'Yearly Income in addition to Salary (e.g. Rental Income)',
    'Profession',
    'Body Height [cm]',
    target
]


def preprocessing(df):
    
    df = df[training_columns]
    
    df['Gender'].replace('0', 'unknown', inplace=True)
    df['Gender'].fillna('unknown', inplace=True)
    
    df['Year of Record'].fillna(df['Year of Record'].median(), inplace=True)
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    df["University Degree"].replace('0', 'No', inplace=True)
    df['University Degree'].fillna('unknown', inplace=True)
    
    df["Housing Situation"].replace('0', 'unknown', inplace=True)
    df["Housing Situation"].replace(0, 'unknown', inplace=True)
    df["Housing Situation"].replace('nA', 'unknown', inplace=True)
    df['Housing Situation'].fillna('unknown', inplace=True)
    
    
    df['Work Experience in Current Job [years]'].replace('#NUM!', '0', inplace=True)
    work_ex = [float(x) for x in df['Work Experience in Current Job [years]']]
    df['Work Experience in Current Job [years]'] = work_ex
    df['Work Experience in Current Job [years]'].fillna(df['Work Experience in Current Job [years]'].median(),
                                                        inplace=True)

    df['Satisfation with employer'].fillna('unknown', inplace=True)
    
    extra = [x.replace(' EUR', '') for x in df['Yearly Income in addition to Salary (e.g. Rental Income)']]
    numerical_extra = [float(x) for x in extra]
    df['Yearly Income in addition to Salary (e.g. Rental Income)'] = numerical_extra
    df['Yearly Income in addition to Salary (e.g. Rental Income)'].fillna(
        df['Yearly Income in addition to Salary (e.g. Rental Income)'].median(), inplace=True)

    df['Crime Level in the City of Employement'].fillna(df['Crime Level in the City of Employement'].median(),
                                                        inplace=True)

    df['Profession'].fillna('unknown', inplace=True)
    
    return df

train = pd.read_csv("D:\PythonProjects\ML_Group_Data/tcd-ml-comp-201920-income-pred-group/train.csv")
test = pd.read_csv("D:\PythonProjects\ML_Group_Data/tcd-ml-comp-201920-income-pred-group/test.csv")

train_data = preprocessing(train)
test_data = preprocessing(test)

y = train_data[target]
train_data.drop(target, axis=1, inplace=True)
test_data.drop(target, axis=1, inplace=True)

enc = TargetEncoder(
    cols=['Gender', 'Country', 'Profession', 'University Degree', 'Housing Situation', 'Satisfation with employer'])
enc.fit(train_data, y)
train_data = enc.transform(train_data)
test_data = enc.transform(test_data)
train_data.head()
test_data.head()

#X_Train, X_Test, y_train, y_test = train_test_split(train_data, y, test_size=0.3, random_state=1)
X_Train = train_data
y_train = y

y_train_log = np.log(y_train)

training = lgb.Dataset(X_Train, y_train_log)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['num_leaves'] = 100
params['min_data'] = 50
params['max_depth'] = 15
params['n_jobs'] = 12
regr = lgb.train(params, training, 200000)
y_prediction = np.exp(regr.predict(X_Test))
y_pred = np.exp(regr.predict(X_Test))

# *********** Submission ***********

instance = np.array(range(1, 369439))
Y_actual_prediction = np.exp(regr.predict(test_data))
sub = pd.DataFrame({'Instance': instance, 'Total Yearly Income [EUR]': Y_actual_prediction})
sub.to_csv('D:\PythonProjects\KaggleGroup/lgbm_second_last.csv', index=False)
