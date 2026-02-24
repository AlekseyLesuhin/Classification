import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import CountEncoder
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
seed = 42

def get_feature_importance(lst: list):
    for i in range(3):
        feature_names = lst[i].named_steps['preprocessor'].get_feature_names_out()
        if i == 0:
            importance = lst[i].named_steps['model'].coef_[0]
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values(by='importance', key=abs, ascending=False)
        else:
            importance = lst[i].named_steps['model'].feature_importances_
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(24, 18))
        sns.barplot(data = feature_importance, x = 'importance', y = 'feature', orient='h')
        ax.set_title(lst[i].named_steps['model'].__class__.__name__, fontsize=18)
        ax.set_xlabel('Values', fontsize=14)
        ax.set_ylabel("Features", fontsize=14)
        ax.tick_params(axis='x', rotation=0, labelsize=12)

def bin_count_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('deposit', axis=1)
    y = df['deposit'].map({'yes': 1, 'no': 0})  # бинаризация таргета
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    cat_cols.remove('day')
    day_cal = ['day']
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    count_pipeline = Pipeline([
        ('count_encoder', CountEncoder()),
        ('scaler', StandardScaler())
    ])
    
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('count', count_pipeline, day_cal)
    ])
    
    # ====== 4. Разбиение с stratify ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y  # разбиенте со стратификацией
    )
    
    # ====== 5. Модели ======
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=seed
        )
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # разбиенте со стратификацией
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = [roc, accuracy, precision, recall, f1]
    
    result_test = pd.DataFrame(results, index=['roc_auc', 'accuracy', 'precision', 'recall', 'f1']).T

    return result_cross_val, result_test, list_of_models

def new_col_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('deposit', axis=1)
    y = df['deposit'].map({'yes': 1, 'no': 0})  # бинаризация таргета
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])  
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # ====== 4. Разбиение с stratify ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y  # разбиенте со стратификацией
    )
    
    # ====== 5. Модели ======
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=seed
        )
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # разбиенте со стратификацией
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = [roc, accuracy, precision, recall, f1]
    
    result_test = pd.DataFrame(results, index=['roc_auc', 'accuracy', 'precision', 'recall', 'f1']).T

    return result_cross_val, result_test, list_of_models

def target_enc_pipe(df):
    # ====== 1. Разделение на X и y ======
    X = df.drop('deposit', axis=1)
    y = df['deposit'].map({'yes': 1, 'no': 0})  # бинаризация таргета
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    cat_cols.remove('duration_backet')
    cat_cols.remove('poutcome')
    cat_cols.remove('education')
    cat_cols.remove('contact')
    te_cals = ['duration_backet', 'poutcome', 'education', 'contact']
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    te_pipeline = Pipeline([
        ('target_encoder', TargetEncoder()),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('te', te_pipeline, te_cals)
    ])
    
    # ====== 4. Разбиение с stratify ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y  # разбиенте со стратификацией
    )
    
    # ====== 5. Модели ======
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=seed
        )
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # разбиенте со стратификацией
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = [roc, accuracy, precision, recall, f1]
    
    result_test = pd.DataFrame(results, index=['roc_auc', 'accuracy', 'precision', 'recall', 'f1']).T

    return result_cross_val, result_test, list_of_models

def outliers_pipe(df):
    
    class QuantileClipper(BaseEstimator, TransformerMixin):
        def __init__(self, lower=0.01, upper=0.99):
            self.lower = lower
            self.upper = upper
            
        def fit(self, X, y=None):
            self.lower_bounds_ = np.quantile(X, self.lower, axis=0)
            self.upper_bounds_ = np.quantile(X, self.upper, axis=0)
            return self
        
        def transform(self, X):
            return np.clip(X, self.lower_bounds_, self.upper_bounds_)

        def get_feature_names_out(self, input_features=None):
            return input_features
    
    # ====== 1. Разделение на X и y ======
    X = df.drop('deposit', axis=1)
    y = df['deposit'].map({'yes': 1, 'no': 0})  # бинаризация таргета
    
    # ====== 2. Определение типов признаков ======
    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(exclude='number').columns.to_list()
    
    # ====== 3. Препроцессинг ======
    num_pipeline = Pipeline([
        ('clipper', QuantileClipper(lower=0.01, upper=0.99)),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])  
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # ====== 4. Разбиение с stratify ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y  # разбиенте со стратификацией
    )
    
    # ====== 5. Модели ======
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(
            eval_metric='logloss',
            random_state=seed
        )
    }
    
    # ====== 6. Кросс-валидация ======
    scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) # разбиенте со стратификацией
    
    cross_val_res = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        cross_val={}
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            cross_val[metric] = str(round(mean_score, 4)) + ' ± ' + str(round(std_score, 4))
        cross_val_res[name] = cross_val
    
    result_cross_val = pd.DataFrame(cross_val_res).T
    
    #====== 7. Финальное обучение и тест ======
    list_of_models=[]
    results={}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = [roc, accuracy, precision, recall, f1]
    
    result_test = pd.DataFrame(results, index=['roc_auc', 'accuracy', 'precision', 'recall', 'f1']).T

    return result_cross_val, result_test, list_of_models



