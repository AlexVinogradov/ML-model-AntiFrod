# Импортируем необходимые библиотеки

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from datetime import datetime
import dill
import numpy as np
from pathlib import Path

# Определяем пути

path = Path('.', 'result_df_final.csv')
models_folder = Path('.', 'models')

# Определим функцию чтения данных

def read_data():

    print('Загружаем данные...')
    data = pd.read_csv(path)
    X = data.drop(['id', 'flag'], axis=1)
    y = data['flag']

    return X, y

# Определим функцию создания новых переменных

def create_features(data):

    # Добавим атрибут, означающий что у клиента нет ни одной просроченной задолженности
    
    data['is_zero_loans'] = data.apply(lambda x: 1.0 if (x.is_zero_loans5_1==1 and 
                                                    x.is_zero_loans530_1==1 and
                                                    x.is_zero_loans3060_1==1 and
                                                    x.is_zero_loans6090_1==1 and
                                                    x.is_zero_loans90_1==1) else 0.0, axis=1)

    # Добавим атрибут, означающий что у клиента нет просрочек и долгов
    
    data['is_zero_debt'] = data.apply(lambda x: 1.0 if (x.is_zero_util_1==1 and 
                                                    x.is_zero_over2limit_1==1 and
                                                    x.is_zero_maxover2limit_1==1) else 0.0, axis=1)    
    return data

# Зададим параметры модели
def pipeline():    
    print("Оценка кредитоспособности клиента...")

    X, y = read_data()

    column_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0))
    ])

    # Создадим pipeline
    preprocessor = Pipeline(steps=[
        ('create_features', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    model = LGBMClassifier(objective = "binary",
                           metric = "auc",
                           learning_rate = 0.05,
                           max_depth = 5,
                           reg_lambda = 1,
                           num_leaves = 64,
                           n_jobs = 5,
                           n_estimators = 2000)
    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print('Обучение модели...')
    
    pipe.fit(X, y)

    print('Сохранение модели...')
    models_folder.mkdir(exist_ok=True)
    filename = 'model_loan_final.pkl'

    with open(models_folder / filename, 'wb') as file:
        dill.dump(pipe, file)       
        
# Запускаем конвейер

pipeline()

print('Загружаем обученную модель...')
with open('models/model_loan_final.pkl', 'rb') as file:
    loan_model = dill.load(file)

print('Обучим модель на всех данных и определим ее точность...')

X, y = read_data()
y_pred = loan_model.predict_proba(X)[:,1]
score = roc_auc_score(y, y_pred)

print('Метрика ROC-AUC на всех данных составляет: ', score)





















