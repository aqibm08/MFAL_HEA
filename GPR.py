import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch.nn.functional as F
from scipy.stats import norm
import seaborn as sns
filterwarnings('ignore')
data = pd.read_csv('Data_base.csv')
data = data.iloc[:, 1:]
sns.set_style('whitegrid') 

X = data[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data[['TEC']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.from_numpy(X_scaled)
y = np.reshape(y, (np.size(y), 1))

y.max()
print("shape of X:", np.shape(X))
print("shape of y:", np.shape(y))
y_tensor = torch.from_numpy(y)

X_train_tensor,X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
gp = SingleTaskGP(X_train_tensor, y_train_tensor)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)


with torch.no_grad():
    posterior = gp.posterior(X_test_tensor)
    mean = posterior.mean.squeeze()
    std = posterior.stddev.squeeze()

y_test_np = y_test_tensor.numpy()
y_pred_np = mean.numpy()

r2 = r2_score(y_test_np, y_pred_np)
mae = mean_absolute_error(y_test_np, y_pred_np)
rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))

print(f"R^2: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

torch.save({
    'model_state_dict': gp.state_dict(),
    'likelihood_state_dict': gp.likelihood.state_dict()
}, "GPR_model.pth")


kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores, mae_scores, rmse_scores = [], [], []

for train_index, test_index in kf.split(X_tensor):
    X_train_fold = X_tensor[train_index]
    X_test_fold = X_tensor[test_index]
    y_train_fold = y_tensor[train_index]
    y_test_fold = y_tensor[test_index]
    
    gp_fold = SingleTaskGP(X_train_fold, y_train_fold)
    mll_fold = ExactMarginalLogLikelihood(gp_fold.likelihood, gp_fold)
    fit_gpytorch_model(mll_fold)
    
    with torch.no_grad():
        posterior_fold = gp_fold.posterior(X_test_fold)
        mean_fold = posterior_fold.mean.squeeze()
    
    y_test_fold_np = y_test_fold.numpy()
    y_pred_fold_np = mean_fold.numpy()
    
    r2_fold = r2_score(y_test_fold_np, y_pred_fold_np)
    mae_fold = mean_absolute_error(y_test_fold_np, y_pred_fold_np)
    rmse_fold = np.sqrt(mean_squared_error(y_test_fold_np, y_pred_fold_np))
    
    r2_scores.append(r2_fold)
    mae_scores.append(mae_fold)
    rmse_scores.append(rmse_fold)

avg_r2 = np.mean(r2_scores)
avg_mae = np.mean(mae_scores)
avg_rmse = np.mean(rmse_scores)

print(f"5-Fold CV - Avg R^2: {avg_r2}")
print(f"5-Fold CV - Avg MAE: {avg_mae}")
print(f"5-Fold CV - Avg RMSE: {avg_rmse}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred_np, alpha=0.7, label=f'R²: {avg_r2:.3f}\nRMSE: {avg_rmse:.3f}\nMAE: {avg_mae:.3f}')
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', label='Ideal Fit')
plt.xlabel("CTE Original", fontsize=14, fontweight='bold')
plt.ylabel("CTE Predicted", fontsize=14, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.savefig("Original_vs_Predicted_GPR.png", dpi = 500)
#plt.show()
plt.show()