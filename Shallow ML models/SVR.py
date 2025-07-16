import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_pickle('./df_aflow_ml_0729.pkl')
X = df.drop(columns=['Egap']).values
y = df["Egap"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    'C': [5000],  
    'epsilon': [0.1],
    'kernel': ['rbf']
}

svr_model = SVR()
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
best_svr_model = grid_search.best_estimator_
y_pred_svr = best_svr_model.predict(X_test)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

n = len(y_test)
mae_uncertainty = mae_svr * np.sqrt(np.pi / (2 * n))
rmse_uncertainty = rmse_svr / np.sqrt(2 * (n - 1))
r2_uncertainty = np.sqrt((4 * r2_svr * (1 - r2_svr)**2) / (n - 2))  

print(f"Mean Absolute Error: {mae_svr:.3f} ± {mae_uncertainty:.3f}")
print(f"Root Mean Squared Error: {rmse_svr:.3f} ± {rmse_uncertainty:.3f}")
print(f"R-squared: {r2_svr:.3f} ± {r2_uncertainty:.3f}")

np.savez('./parity_data_svr.npz', true_labels = np.array(y_test),
              predicted_labels = np.array(y_pred_svr))

