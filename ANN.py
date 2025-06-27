import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA
import pickle
data = pd.read_csv('Data_base.csv')
data = data.iloc[:, 1:]
sns.set_style('whitegrid') 

X = data[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data[['TEC']]
var = X.var()
zero_var_columns = var[var == 0].index.to_list()

zero_var_columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
contamination = 0.05
iso_forest = IsolationForest(contamination=contamination, random_state=42)
outliers_iso = iso_forest.fit_predict(X_scaled)
outliers_data_iso = data[outliers_iso == -1]
#print(f"Isolation Forest - {len(outliers_data_iso)}")

iso_outliers_set = set(outliers_data_iso.index)
data_cleaned = data.drop(index=list(iso_outliers_set))

X = data_cleaned[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data_cleaned[['TEC']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y.values.ravel(), test_size=0.2, random_state=42)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super(NNRegressor, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i]))  #batch norma added for stabiliy
            layers.append(nn.Dropout(dropout_rate))         # dropout added to prevent overfitting
        layers.append(nn.Linear(hidden_dims[-1], 1))        # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


criterion = nn.MSELoss()  # MSE Loss

hidden_dims = [256, 256, 256, 64, 32]
dropout_rate = 0.05
lr = 0.003   #learning rate
weight_decay = 1e-4  # regularization
batch_size = 64
epochs = 100

model = NNRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.3)

#dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
best_r2 = float('-inf')
training_losses = []
validation_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_batch = model(X_batch)
        loss = criterion(y_pred_batch, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    training_losses.append(epoch_loss / len(train_loader))
    #evaluating model
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_test_tensor)
        val_loss = criterion(y_pred_val, y_test_tensor).item()
        validation_losses.append(val_loss)
        r2 = r2_score(y_test_tensor.numpy(), y_pred_val.numpy())
        scheduler.step(val_loss)

        if r2 > best_r2:
            best_r2 = r2
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping condition
       # if patience_counter >= early_stopping_patience:
       #     print(f"Early stopping at epoch {epoch + 1}")
       #     break

plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig("Training_validation_loss.png", dpi = 400)
print(f"Final R²: {best_r2:.4f}")
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    r2 = r2_score(y_test_tensor.numpy(), y_pred_test.numpy())
    mse = mean_squared_error(y_test_tensor.numpy(), y_pred_test.numpy())
    mae = mean_absolute_error(y_test_tensor.numpy(), y_pred_test.numpy())

print(f"Enhanced Model Test R²: {r2:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_tensor.numpy(), y_pred_test.numpy(), c='#6fffe9', edgecolors='black', linewidths=0.7, alpha=0.7, label=f'R²: {r2:.3f}\nRMSE: {np.sqrt(mse):.3f}\nMAE: {mae:.3f}')
plt.plot([y_test_tensor.min(), y_test_tensor.max()], [y_test_tensor.min(), y_test_tensor.max()], '--', color='red')
plt.xlabel("CTE Original", fontsize=14, fontweight='bold')
plt.ylabel("CTE Predicted", fontsize=14, fontweight='bold')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, facecolor='white')
plt.savefig("Original_vs_Predicted_NN_f.png", dpi = 500)
plt.show()
#model_file = "nn_regressor.pkl"

#with open(model_file, "wb") as f:
#    pickle.dump(model.state_dict(), f)