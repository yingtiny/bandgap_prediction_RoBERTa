import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

df = pd.read_pickle('./df_aflow_ml_0729.pkl')


X = df.drop(columns=['Egap']).values
y = df["Egap"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [2000],  
    'max_depth': [9],        
    'learning_rate': [ 0.1],
    'subsample': [1.0],     
    'colsample_bytree': [0.6] 
}


xgb_model = XGBRegressor(objective='reg:squarederror', random_state=84)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

best_xgb_model = grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

n = len(y_test)
mae_uncertainty = mae_xgb * np.sqrt(np.pi / (2 * n))
rmse_uncertainty = rmse_xgb / np.sqrt(2 * (n - 1))
r2_uncertainty = np.sqrt((4 * r2_xgb * (1 - r2_xgb)**2) / (n - 2)) 

print(f"Mean Absolute Error: {mae_xgb:.3f} ± {mae_uncertainty:.3f}")
print(f"Root Mean Squared Error: {rmse_xgb:.3f} ± {rmse_uncertainty:.3f}")
print(f"R-squared: {r2_xgb:.3f} ± {r2_uncertainty:.3f}")

np.savez('./parity_data_xg.npz', true_labels = np.array(y_test),
              predicted_labels = np.array(y_pred_xgb))