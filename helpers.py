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

def run_experiment_pipeline(
    df,
    *,
    seed=42,
    test_size=0.2,
    # preprocessing configuration (all keys optional)
    num_cols=None,            # list or None -> auto detect numeric
    cat_cols=None,            # list or None -> auto detect categorical
    onehot_cols=None,         # subset of cat_cols to one-hot encode (default = all cat_cols)
    count_cols=None,          # list of columns to CountEncode
    target_cols=None,         # list of columns to TargetEncode
    clip_cols=None,           # list of numeric columns to apply QuantileClipper
    clip_quantiles=(0.01, 0.99),
    drop_cols=None,           # list of columns to drop from X before processing
    # models and scoring
    models=None,              # dict name->estimator, default same as original
    scoring=None,             # list or None -> default ['roc_auc','accuracy','precision','recall','f1']
    n_splits=5,
    n_jobs=-1
):
    """
    Универсальная функция для тестирования разных препроцессингов и моделей.
    Возвращает (result_cross_val, result_test, list_of_models).
    """

    # ===== 1. X, y =====
    if 'deposit' not in df.columns:
        raise ValueError("Входной df должен содержать колонку 'deposit'")
    X = df.drop('deposit', axis=1).copy()
    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')
    y = df['deposit'].map({'yes': 1, 'no': 0})

    # ===== 2. Автовыбор колонок =====
    num_cols_all = X.select_dtypes(include='number').columns.to_list()
    cat_cols_all = X.select_dtypes(exclude='number').columns.to_list()

    if num_cols is None:
        num_cols = num_cols_all.copy()
    else:
        # keep only existing
        num_cols = [c for c in num_cols if c in X.columns]

    if cat_cols is None:
        cat_cols = cat_cols_all.copy()
    else:
        cat_cols = [c for c in cat_cols if c in X.columns]

    # default onehot = all categorical except those assigned to other encoders
    if onehot_cols is None:
        onehot_cols = cat_cols.copy()
    else:
        onehot_cols = [c for c in onehot_cols if c in cat_cols]

    # ensure lists exist
    count_cols = [c for c in (count_cols or []) if c in X.columns]
    target_cols = [c for c in (target_cols or []) if c in X.columns]
    clip_cols = [c for c in (clip_cols or []) if c in X.columns]

    # remove special columns from onehot / generic cat list
    # columns assigned to count or target should not be one-hot encoded
    for c in count_cols + target_cols:
        if c in onehot_cols:
            onehot_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    # if clip_cols in numeric, remove them from generic numeric pipeline (they'll be handled separately)
    num_cols_for_standard = [c for c in num_cols if c not in clip_cols]

    # ===== 3. Build pipelines =====
    transformers = []

    # numeric pipeline (standard scaler)
    if len(num_cols_for_standard) > 0:
        num_pipeline = Pipeline([('scaler', StandardScaler())])
        transformers.append(('num', num_pipeline, num_cols_for_standard))

    # clipper pipeline for selected numeric columns
    if len(clip_cols) > 0:
        clip_pipeline = Pipeline([
            ('clipper', QuantileClipper(lower=clip_quantiles[0], upper=clip_quantiles[1])),
            ('scaler', StandardScaler())
        ])
        transformers.append(('clip', clip_pipeline, clip_cols))

    # one-hot for remaining categorical
    if len(onehot_cols) > 0:
        cat_pipeline = Pipeline([
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        transformers.append(('cat_ohe', cat_pipeline, onehot_cols))

    # count encoder pipelines (each column via CountEncoder -> scaler)
    if len(count_cols) > 0:
        # CountEncoder can handle multiple cols at once; but we'll apply one transformer for all count_cols
        count_pipeline = Pipeline([
            ('count_enc', CountEncoder(cols=count_cols)),
            ('scaler', StandardScaler())
        ])
        # CountEncoder expects DataFrame input for those columns; give the list
        transformers.append(('count', count_pipeline, count_cols))

    # target encoder pipeline
    if len(target_cols) > 0:
        te_pipeline = Pipeline([
            ('te', TargetEncoder(cols=target_cols)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('target', te_pipeline, target_cols))

    if len(transformers) == 0:
        raise ValueError("Нечего трансформировать — проверьте конфигурацию колонок.")

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # ===== 4. Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ===== 5. Models =====
    if models is None:
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=seed),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=seed)
        }

    if scoring is None:
        scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ===== 6. Cross-validation =====
    cross_val_res = {}
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs, error_score='raise'
        )
        cross_val = {}
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            cross_val[metric] = round(mean_score, 4)
            # if you want ± std, you can store as f"{mean:.4f} ± {std:.4f}"
        cross_val_res[name] = cross_val

    result_cross_val = pd.DataFrame(cross_val_res).T

    # ===== 7. Fit on full train and evaluate on test =====
    list_of_models = []
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        list_of_models.append(pipeline)

        y_pred = pipeline.predict(X_test)
        # some estimators may not have predict_proba; try/except
        try:
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            # fallback: use decision_function if available, else set roc = np.nan
            try:
                scores = pipeline.decision_function(X_test)
                roc = roc_auc_score(y_test, scores)
            except Exception:
                roc = np.nan

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name] = [roc, accuracy, precision, recall, f1]

    result_test = pd.DataFrame(results, index=['roc_auc', 'accuracy', 'precision', 'recall', 'f1']).T

    return result_cross_val, result_test, list_of_models





