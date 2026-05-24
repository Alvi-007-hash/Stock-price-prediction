import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,VotingRegressor,StackingRegressor,GradientBoostingRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
import pickle
warnings.filterwarnings('ignore')
df = pd.read_csv(r'C:\Users\RAKIB\OneDrive\Desktop\ml & dl\projects\stock_data.csv')
print(df.head())
df.rename(columns={'Unnamed: 0': 'date'},inplace=True)
print(df.head())
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df.sort_values('date',inplace=True)
print(df.head())
df['Stock_1_lag_1'] = df['Stock_1'].shift(1)
df['Stock_1_lag_2'] = df['Stock_1'].shift(2)
df['Stock_1_mae5'] = df['Stock_1'].rolling(window=5).mean()
df['Stock_2_lag_1'] = df['Stock_2'].shift(1)
df['Stock_3_lag_1'] = df['Stock_3'].shift(1)
df['Stock_4_lag_1'] = df['Stock_4'].shift(1)
df['Stock_5_lag_1'] = df['Stock_5'].shift(1)
df['portfolio_mean'] = df[['Stock_1','Stock_2','Stock_3','Stock_4','Stock_5']].mean(axis=1)
df['portfolio_std'] = df[['Stock_1','Stock_2','Stock_3','Stock_4','Stock_5']].std(axis=1)
df.dropna(inplace=True)
df['target'] = df['Stock_1'].shift(-1)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.dropna(axis=1,inplace=True)
df.drop('date',axis=1,inplace=True)
df.drop('year',axis=1,inplace=True)
print(df.head())
feature_cols = ['month','day','day_of_week','day_of_year','Stock_1_lag_1','Stock_1_lag_2','Stock_1_mae5','Stock_2_lag_1',
                'Stock_3_lag_1','Stock_4_lag_1','Stock_5_lag_1','portfolio_mean','portfolio_std'

                ]
x = df[feature_cols]
y = df['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
numeric_cols = x.columns.to_list()
numeric_transformer  = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())]
)
preprocessor = ColumnTransformer(
    transformers = [
        ('num',numeric_transformer,numeric_cols)
    ]
)
rf = RandomForestRegressor(n_estimators=100,random_state=42)
xg_model = xgb.XGBRegressor(n_estimators=100,random_state=42)
ridge = Ridge(alpha=0.1)
gb = GradientBoostingRegressor(n_estimators=100,random_state=42)

voting = VotingRegressor(
    estimators = [
        ('rf',rf),
        ('xg',xg_model),
        ('ridge',ridge),
        ('gb',gb)
    ]
)
staking = StackingRegressor(
    estimators = [
        ('rf',rf),
        ('xg',xg_model),
        ('ridge',ridge),
        ('gb',gb)
    ],
    final_estimator = Ridge()
)
model_to_train = {
    'rf':rf,
    'xg':xg_model,
    'ridge':ridge,
    'gb':gb,
    'voting':voting,
    'stacking':staking
}


results = []
for name,model in model_to_train.items():
  pipe = Pipeline(steps=[
      ('preprocessor',preprocessor),
      ('model',model)
  ])
  pipe.fit(x_train,y_train)
  y_pred_test = pipe.predict(x_test)
  y_pred_train = pipe.predict(x_train)
  results.append({
      'model':name,
      'test_mse':mean_squared_error(y_test,y_pred_test),
      'test_r2':r2_score(y_test,y_pred_test),
      'train_mse':mean_squared_error(y_train,y_pred_train),
      'train_r2':r2_score(y_train,y_pred_train)
  })

results_df = pd.DataFrame(results)
results_df['overfitting_gap'] = results_df['train_r2'] - results_df['test_r2']
results_df.sort_values('overfitting_gap',ascending=True,inplace=True)
results_df.reset_index(drop=True,inplace=True)
print(results_df)
best_model_name = results_df.iloc[0]['model']
best_model_obj = model_to_train[best_model_name]
print(best_model_name)
final_pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',best_model_obj)
])
final_pipe.fit(x_train,y_train)
y_pred_test = final_pipe.predict(x_test)
print(f'test_mse: {mean_squared_error(y_test,y_pred_test)}')
cv_score = cross_val_score(
    final_pipe,
    x_train,
    y_train,
    cv = 5,
    scoring = 'r2'
)
print(cv_score)
print(cv_score.mean())
param_grid = {
    'ridge' : {
        'model__alpha':[0.001,0.01,0.1,1,10,100]
    },
    'rf':{
        'model__n_estimators':[100,200,300],
        'model__max_depth':[None,10,20,30],
        'model__min_samples_split':[2,5,10],
        'model__min_samples_leaf':[1,2,4]
    },
    'xg':{
        'model__n_estimators':[100,200,300],
        'model__learning_rate':[0.01,0.1,0.2],
        'model__max_depth':[3,4,5]
    },
    'gb':{
        'model__n_estimators':[100,200,300],
        'model__learning_rate':[0.01,0.1,0.2],
        'model__max_depth':[3,4,5]
    },
    'voting':{
        'model__rf__n_estimators':[100,200,300],
        'model__rf__max_depth':[None,10,20,30],
        'model__rf__min_samples_split':[2,5,10],
        'model__rf__min_samples_leaf':[1,2,4]
    },
    'stacking':{
        'model__final_estimator__alpha':[0.001,0.01,0.1,1,10,100]
    }
}
grid_search = GridSearchCV(
    estimator = final_pipe,
    param_grid = param_grid[best_model_name],
    cv = 5,
    scoring = 'r2',
    n_jobs = -1,
    verbose = 2
)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(x_test)
y_pred_traning = best_model.predict(x_train)
print(f'test_r2: {r2_score(y_test,y_pred_test)}')
print(f'test_mse: {mean_squared_error(y_test,y_pred_test)}')
print(f'train_r2: {r2_score(y_train,y_pred_traning)}')
print(f'train_mse: {mean_squared_error(y_train,y_pred_traning)}')
file_name = 'stock_model.pkl'
with open(file_name,'wb') as f:
  pickle.dump(best_model,f)