import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_pickle('./df_aflow_ml_0729.pkl')


X = df.drop(columns=['Egap']).values
y = df["Egap"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [1000],  
    'max_depth': [50],       
    'min_samples_split': [2], 
    'min_samples_leaf': [2]   
}


rf_model = RandomForestRegressor(random_state=84)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)


grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Hyperparameters={best_params}")

best_rf_model = grid_search.best_estimator_
y_pred_rf_1 = best_rf_model.predict(X_test)

mae_rf_1 = mean_absolute_error(y_test, y_pred_rf_1)
rmse_rf_1 = np.sqrt(mean_squared_error(y_test, y_pred_rf_1))
r2_rf_1 = r2_score(y_test, y_pred_rf_1)

n = len(y_test)
mae_uncertainty = mae_rf_1 * np.sqrt(np.pi / (2 * n))
rmse_uncertainty = rmse_rf_1 / np.sqrt(2 * (n - 1))
r2_uncertainty = np.sqrt((4 * r2_rf_1 * (1 - r2_rf_1)**2) / (n - 2))  


print(f"Mean Absolute Error: {mae_rf_1:.3f} ± {mae_uncertainty:.3f}")
print(f"Root Mean Squared Error: {rmse_rf_1:.3f} ± {rmse_uncertainty:.3f}")
print(f"R-squared: {r2_rf_1:.3f} ± {r2_uncertainty:.3f}")

np.savez('./parity_data_rf.npz', true_labels = np.array(y_test),
              predicted_labels = np.array(y_pred_rf_1))
