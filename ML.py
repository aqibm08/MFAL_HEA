import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score, mean_squared_error, get_scorer_names
from statsmodels.stats.diagnostic import lilliefors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import RFE 
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
sns.set_style('whitegrid') 

np.int = int
data = pd.read_csv("Data_base.csv") #dataset Rao et al
#data.describe()
data = data.iloc[:, 1:]

X = data[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data[['TEC']]
var = X.var()
zero_var_columns = var[var == 0].index.to_list()

zero_var_columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


contamination = 0.05 #contamination
iso_forest = IsolationForest(contamination=contamination, random_state=42) #IF
outliers_iso = iso_forest.fit_predict(X_scaled)
outliers_data_iso = data[outliers_iso == -1]
print(f"Isolation Forest - {len(outliers_data_iso)}")

lof = LocalOutlierFactor(n_neighbors=15, contamination=contamination) #LOF
outliers_lof = lof.fit_predict(X_scaled)
outliers_data_lof = data[outliers_lof == -1]
print(f"LOF - {len(outliers_data_lof)}")

svm = OneClassSVM(kernel="rbf", gamma="auto", nu=contamination) #One class SVM
outliers_svm = svm.fit_predict(X_scaled)
outliers_data_svm = data[outliers_svm == -1]
print(f"One-Class SVM - {len(outliers_data_svm)}")


iso_outliers_set = set(outliers_data_iso.index)
lof_outliers_set = set(outliers_data_lof.index)
svm_outliers_set = set(outliers_data_svm.index)

common_outliers = iso_outliers_set.intersection(lof_outliers_set, svm_outliers_set)
print(f"Outlier deCTEted by all three - {len(common_outliers)} they are{common_outliers}")

#tsne = TSNE(n_components=2, random_state=42)
#X_tsne = tsne.fit_transform(X_scaled)
#
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color='#ADB5BD', edgecolors='black',linewidths= 0.6, alpha=0.6, label='Inliers')
#
#plt.scatter(X_tsne[list(iso_outliers_set), 0], X_tsne[list(iso_outliers_set), 1],linewidths= 0.6, color='#6A0DAD',edgecolors='black', label='Isolation Forest Outliers')
#plt.scatter(X_tsne[list(lof_outliers_set), 0], X_tsne[list(lof_outliers_set), 1],linewidths= 0.6, color='#008080',edgecolors='black', label='LOF Outliers')
#plt.scatter(X_tsne[list(svm_outliers_set), 0], X_tsne[list(svm_outliers_set), 1],linewidths= 0.6, color='#40E0D0', edgecolors='black', label='One-Class SVM Outliers')
#
#plt.scatter(X_tsne[list(common_outliers), 0], X_tsne[list(common_outliers), 1], color='black', label='Common Outliers', marker='x')
#
#plt.legend()
#plt.savefig('tsne_plot_with_outliers.png', dpi = 400)
all_outliers = set.union(iso_outliers_set, lof_outliers_set, svm_outliers_set)

data_cleaned = data.drop(index=list(iso_outliers_set)) #IF outliers removed

X = data_cleaned[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data_cleaned[['TEC']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled) #PCA transformation

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.2, random_state=42)

X_concat = pd.concat([pd.DataFrame(X_test), pd.DataFrame(X_train)], axis=0)
tsne = TSNE(n_components=2, random_state=1)
original_tsne = tsne.fit_transform(X_concat)

n_test = len(X_test)
test_tsne = original_tsne[:n_test]
train_tsne = original_tsne[n_test:]

plt.figure()
plt.scatter(original_tsne[:, 0], original_tsne[:, 1], color='grey', label='Original Data')
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], color='#953553', edgecolor='black', linewidth=0.5, alpha=0.75, label='Train Data')

plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.tight_layout()

plt.savefig('tsne_train_plot.png', dpi=500)
plt.show()
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)


param_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0, prior='uniform')
}
#print(get_scorer_names())
bayes_search = BayesSearchCV(
    estimator=random_forest_model,
    search_spaces=param_space,
    n_iter=50,
    scoring='r2',

    cv=5,
    n_jobs=-1,
    verbose=0,
    random_state=42
)
bayes_search.fit(X_train, y_train)
print("Best Parameters:", bayes_search.best_params_)
best_rf_model = bayes_search.best_estimator_
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='r2')

print("Cross-Validation R² Scores:", cv_scores)
print("Mean CV R² Score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test R²: {r2:.3f}, Test MSE: {mse:.3f}, Test MAE: {mae:.3f}")
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]


#plt.figure(figsize=(10, 6))
#plt.plot(cv_scores, marker='o', color='b')
##plt.title("Cross-Validation R² Scores")
#plt.xlabel("Fold")
#plt.ylabel("R² Score")
#plt.grid()
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='#3DAC78', edgecolors='black', linewidths=0.7, alpha=0.7, label=f'R²: {r2:.3f}\nRMSE: {np.sqrt(mse):.3f}\nMAE: {mae:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("CTE Original", fontsize=14, fontweight='bold')
plt.ylabel("CTE Predicted", fontsize=14, fontweight='bold')
#plt.title("Original vs Predicted: Random Forest", fontsize=16, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.savefig("Original_vs_Predicted_Random_Forest.png", dpi = 500)
#plt.show()

xgboost_model = XGBRegressor(objective='reg:squarederror', random_state=42)
# parameter space
param_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(3, 20),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5),
    'min_child_weight': Integer(1, 10)
}
# tuning hyperparameters
bayes_search = BayesSearchCV(
    estimator=xgboost_model,
    search_spaces=param_space,
    n_iter=50,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=0,
    random_state=42
)
bayes_search.fit(X_train, y_train)
print("Best Parameters:", bayes_search.best_params_)
best_xgb_model = bayes_search.best_estimator_

cv_scores = cross_val_score(best_xgb_model, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R² Scores:", cv_scores)
print("Mean CV R² Score:", np.mean(cv_scores))

y_pred = best_xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test R²: {r2:.3f}, Test MSE: {mse:.3f}, Test MAE: {mae:.3f}")

#plt.figure(figsize=(10, 6))
#plt.plot(cv_scores, marker='o', color='b')
##plt.title("Cross-Validation R² Scores")
#plt.xlabel("Fold")
#plt.ylabel("R² Score")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='#91e5f6', edgecolors='black', linewidths=0.7, alpha=0.7, label=f'R²: {r2:.3f}\nRMSE: {np.sqrt(mse):.3f}\nMAE: {mae:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("CTE Original", fontsize=14, fontweight='bold')
plt.ylabel("CTE Predicted", fontsize=14, fontweight='bold')
#plt.title("Original vs Predicted: XGBoost", fontsize=16, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.savefig("Original_vs_Predicted_XGBoost.png", dpi = 500)

#plt.show()


svr_model = SVR() # SVR

# parameter search space for Bayesian optimization
param_space = {
    'C': Real(1e-3, 1e3, prior='log-uniform'),
    'epsilon': Real(0.01, 1.0, prior='log-uniform'),
    'gamma': Real(1e-4, 1, prior='log-uniform'),
    'kernel': ['rbf']  # Using 'rbf' as it is commonly used for regression
}

# bayesian search for hyperparameter tuning
bayes_search = BayesSearchCV(
    estimator=svr_model,
    search_spaces=param_space,
    n_iter=50,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=0,
    random_state=42
)

# fit Bayesian optimization
bayes_search.fit(X_train, y_train)
print("Best Parameters:", bayes_search.best_params_)

best_svr_model = bayes_search.best_estimator_

cv_scores = cross_val_score(best_svr_model, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R² Scores:", cv_scores)
print("Mean CV R² Score:", np.mean(cv_scores))

y_pred = best_svr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test R²: {r2:.3f}, Test MSE: {mse:.3f}, Test MAE: {mae:.3f}")

#plt.figure(figsize=(10, 6))
#plt.plot(cv_scores, marker='o', color='b')
##plt.title("Cross-Validation R² Scores for SVR")
#plt.xlabel("Fold")
#plt.ylabel("R² Score")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='#0077B6', edgecolors='black', linewidths=0.7, alpha=0.7, label=f'R²: {r2:.3f}\nRMSE: {np.sqrt(mse):.3f}\nMAE: {mae:.3f}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("CTE Original", fontsize=14, fontweight='bold')
plt.ylabel("CTE Predicted", fontsize=14, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.savefig("Original_vs_Predicted_SVR.png", dpi = 500)

plt.show()


