import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
kaggle_train = '../input/house-prices-advanced-regression-techniques/train.csv'
kaggle_test  = '../input/house-prices-advanced-regression-techniques/test.csv'
local_train = 'train.csv'
local_test  = 'test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test_ids = test['Id'].copy()

y_orig = train['SalePrice'].copy()
y = np.log1p(y_orig)  
X = train.drop(columns=['SalePrice', 'Id'])
X_test = test.drop(columns=['Id'])
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='drop', sparse_threshold=0)
model = Pipeline(steps=[
    ('preproc', preprocessor),
    ('ridge', Ridge(alpha=10, random_state=42))
])
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores)
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)  
preds[preds < 0] = 0.0
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
output_path = 'submission.csv'
submission.to_csv(output_path, index=False)

