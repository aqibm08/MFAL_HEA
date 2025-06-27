import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import torch
import time
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from botorch.exceptions.errors import ModelFittingError
from botorch.models import MultiTaskGP
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
import torch.nn.functional as F
from scipy.stats import norm
import math
import gpytorch
import seaborn as sns
filterwarnings('ignore')
data = pd.read_csv('Data_base.csv')
data = data.iloc[:, 1:]
sns.set_style('whitegrid')


X = data[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
y = data[['TEC']]
pos_mask = y.values.flatten() >= 0
X = X.iloc[pos_mask]
y = y.iloc[pos_mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=5)
X_transformed = pca.fit_transform(X_scaled)



X_tensor = torch.from_numpy(X_transformed)
y = np.reshape(y, (np.size(y), 1))



y_tensor = torch.from_numpy(y)
fidelity_costs = [1,5]
acquisition_methods = ['EI'] # can be anything
nruns = 100
niter =200
nsamp = 20


X_train_tensor,X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.7, random_state=42)
middle_fidelity_model = SingleTaskGP(X_train_tensor, y_train_tensor)
mll = ExactMarginalLogLikelihood(middle_fidelity_model.likelihood, middle_fidelity_model)
fit_gpytorch_mll(mll)


def augment_input_with_fidelity(X, fidelity_level):
    return torch.cat([X, fidelity_level], dim=-1)
fidelity_low = torch.zeros(X_tensor.size(0), 1)
fidelity_high = torch.ones(X_tensor.size(0), 1)
X_low_augmented = augment_input_with_fidelity(X_tensor, fidelity_low)
X_high_augmented = augment_input_with_fidelity(X_tensor, fidelity_high)

with torch.no_grad():
    posterior = middle_fidelity_model.posterior(X_tensor)
    y_low = posterior.mean.reshape(-1, 1)


X_mogp = torch.cat([X_low_augmented, X_high_augmented], dim=0)

y_mogp = torch.cat([y_low, y_tensor], dim=0)
def knowledge_gradient(mean, var, best_value, fidelity=1):
    """
    calculate knowledge gradient - the expected improvement in the best value
    if we were to sample at a particular point and fidelity.
    """
    std = torch.sqrt(var)
    z = (best_value - mean) / (std + 1e-9)
    return std * (z * norm.cdf(z.numpy()) + norm.pdf(z.numpy()))

def efficient_cost_aware_acquisition(mean, var, best_value, iteration, max_iterations, 
                              cost_ratio=5.0, temperature=0.5, epsilon=0.01):
    """    
    Args:
        mean: Tensor of predicted means with shape [num_points, 2], where 2nd dim is fidelity
        var: Tensor of predicted variances with shape [num_points, 2]
        best_value: Best observed value so far (minimum for minimization problems)
        iteration: Current iteration number
        max_iterations: Maximum number of iterations
        cost_ratio: Ratio of cost between high and low fidelity (high/low)
        temperature: Temperature parameter for softmax (lower = more exploitative)
        epsilon: Small value for numerical stability
    
    Returns:
        selected_point_idx: Index of the selected point
        selected_fidelity: Selected fidelity level (0 for low, 1 for high)
    """
    num_points = mean.shape[0]
    std = torch.sqrt(var)
    
    # calculate progress-dependent parameters
    progress = min(1.0, iteration / max_iterations)
    
    # more aggressive exploitation 
    beta = 2.0 * (1.0 - 0.9 * progress)  # Decreased beta for better exploitation
    
    # adaptive cost penalty: less penalty at start, more at end
    # less aggressive penalty to allow high fidelity when needed
    adaptive_cost_ratio = cost_ratio * (0.5 + 0.5 * progress)
    
    acq_values = torch.zeros(num_points, 2)
    
   
    for fid in range(2):
        # calculate Expected Improvement specifically for minimization
        improvement = best_value - mean[:, fid]
        z = improvement / (std[:, fid] + 1e-9)
        cdf = torch.tensor(norm.cdf(z.numpy()))
        pdf = torch.tensor(norm.pdf(z.numpy()))
        ei = improvement * cdf + std[:, fid] * pdf
        ei = torch.where(ei < 0, torch.zeros_like(ei), ei)  # ensure non-negative
        
        # knowledge gradient component 
        kg = std[:, fid] * (z * cdf + pdf)
        
        # LCB component for minimization (helps find lower values)
        lcb = mean[:, fid] - beta * std[:, fid]
        
        if progress < 0.25:
            # early stage: balance exploration and exploitation
            ei_weight = 0.5
            kg_weight = 0.1
            lcb_weight = 0.4
        elif progress < 0.5:
            # mid-early stage: shift toward more exploitation
            ei_weight = 0.6
            kg_weight = 0.1
            lcb_weight = 0.3
        elif progress < 0.75:
            # mid-late stage: even more exploitation focus
            ei_weight = 0.6
            kg_weight = 0.15
            lcb_weight = 0.25
        else:
            # late stage: heavy exploitation for convergence
            ei_weight = 0.8
            kg_weight = 0.1
            lcb_weight = 0.1
        
        acq_values[:, fid] = ei_weight * ei + kg_weight * kg - lcb_weight * lcb
    
    # apply cost adjustment to high fidelity - but don't penalize too heavily
    # to maintain speed of convergence
    acq_values[:, 1] = acq_values[:, 1] / adaptive_cost_ratio
    
    # first identify the top candidates in each fidelity for more targeted selection
    top_k = max(3, int(num_points * 0.05))  # At least 3 or 5% of points
    
    # get top candidates for each fidelity
    _, top_low_indices = torch.topk(acq_values[:, 0], min(top_k, num_points))
    _, top_high_indices = torch.topk(acq_values[:, 1], min(top_k, num_points))
    
    # two-step selection process: first select fidelity based on budget and progress,
    # then select best point within that fidelity
    
    # Step 1: Determine fidelity based on budget constraints and progress
    # early stage: Allow more high fidelity for better model training
    # late stage: Be more budget conscious but still allow high fidelity for convergence
    high_fid_threshold = 0.7 - 0.4 * progress  # Decreases from 0.7 to 0.3 over iterations
    
    # additional boost for high fidelity in very early stage for model calibration
    if progress < 0.1:
        high_fid_threshold += 0.15
    
    # boost high fidelity at the very end for final convergence
    if progress > 0.9:
        high_fid_threshold += 0.15
    
    if torch.rand(1).item() < high_fid_threshold:
        # high fidelity: select best candidate
        selected_fidelity = 1
        best_high_idx = top_high_indices[0]  # Best high fidelity candidate
        selected_point_idx = best_high_idx
        
        # additional check: if high fidelity doesn't provide significant improvement,
        # fall back to low fidelity to save budget
        best_low_value = mean[top_low_indices[0], 0]
        best_high_value = mean[best_high_idx, 1]
        if best_high_value > best_low_value * 0.95:  # High fidelity not much better
            improvement_ratio = (best_value - best_high_value) / (best_value - best_low_value + epsilon)
            if improvement_ratio < 1.2:  # Not enough improvement to justify high fidelity
                selected_fidelity = 0
                selected_point_idx = top_low_indices[0]
    else:
        # low fidelity: select best candidate
        selected_fidelity = 0
        selected_point_idx = top_low_indices[0]
    
    # special case: Override for situations where we really need high fidelity
    # this helps maintain fast convergence in critical situations
    if progress > 0.5 and selected_fidelity == 0:
        # Late stage + we selected low fidelity
        best_high_idx = top_high_indices[0]
        best_high_acq = acq_values[best_high_idx, 1] * adaptive_cost_ratio  # Undo cost penalty
        best_low_acq = acq_values[selected_point_idx, 0]
        
        # ff high fidelity acquisition is much better, override to high fidelity
        # this maintains convergence speed when critical
        if best_high_acq > best_low_acq * 2.0:
            selected_fidelity = 1
            selected_point_idx = best_high_idx
    
    return selected_point_idx, selected_fidelity


def rapid_convergence_mfal_iteration(X, y, nsamp, niter, cost_ratio=5.0):
    """
    Multi-fidelity active learning optimized for rapid convergence to minima
    while maintaining budget awareness.
    
    Args:
        X: Input features with fidelity in last column
        y: Target values
        nsamp: Initial sample size
        niter: Total iterations
        cost_ratio: Cost ratio between high and low fidelity
    """
    X = X.float()
    y = y.float()
    log = []
    
    total_budget = cost_ratio * (niter - nsamp) * 0.8
    spent_budget = 0
    
    n_high = int(nsamp * 0.4)  # 40% high fidelity samples initially for better model initialization
    n_low = nsamp - n_high     # 60% low fidelity samples
    
    low_fidelity_indices = torch.where(X[:, -1] == 0)[0]
    high_fidelity_indices = torch.where(X[:, -1] == 1)[0]
    
    low_indices = np.random.choice(low_fidelity_indices.numpy(), size=n_low, replace=False)
    high_indices = np.random.choice(high_fidelity_indices.numpy(), size=n_high, replace=False)
    
    get_index = torch.tensor(np.concatenate([low_indices, high_indices]), dtype=torch.long)
    
    # Track fidelity selections for statistics
    low_fid_count = n_low
    high_fid_count = n_high
    
    spent_budget = n_high * cost_ratio + n_low * 1.0
    
    # track best value for early stopping
    best_value_overall = float('inf')
    best_value_patience = 0
    patience_limit = 200  # Stop after 20 iterations without improvement
    
    # early phase high-fidelity boost 
    early_phase = True
    early_phase_limit = min(20, int(0.1 * (niter - nsamp)))  # Early phase is first 20 iterations
    
    for i in range(nsamp, niter):
        remaining_budget = total_budget - spent_budget
        
        if remaining_budget <= 0:
            print(f"Budget exhausted at iteration {i}. Stopping early.")
            break
            
        X_train = X[get_index]
        y_train = y[get_index]
        
        current_best = y_train.min().item()
        
        # early stopping if we've found a good minima and aren't improving
        if current_best < best_value_overall:
            best_value_overall = current_best
            best_value_patience = 0
        else:
            best_value_patience += 1
            
        # early stopping if we've reached convergence
        if best_value_patience > patience_limit and i > nsamp + 50:  # At least 50 iterations
            print(f"Early stopping at iteration {i}: No improvement for {patience_limit} iterations.")
            break
        
        try:
            model = MultiTaskGP(X_train, y_train, task_feature=-1)
            model.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=X_train.shape[1]-1)
        )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            print(f"Warning: Model fitting failed at iteration {i}. Stopping early.")
            return get_index, log, {'low': low_fid_count, 'high': high_fid_count, 'budget': spent_budget}
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_unique = X[:len(y) // 2, :-1]  # Features without fidelity indicator
            posterior = model.posterior(X_unique)
            mean = posterior.mean
            var = posterior.variance
        
        # calculate progress as a fraction of iterations
        progress = (i - nsamp) / (niter - nsamp)
        
        # calculate remaining budget ratio
        budget_ratio = remaining_budget / total_budget
        
        # check if we're in early phase (more aggressive exploration)
        if i - nsamp < early_phase_limit:
            early_phase = True
        else:
            early_phase = False
        
        # adjust cost ratio based on remaining budget, but less aggressively
        effective_cost_ratio = cost_ratio * (1.0 + max(0, 0.3 - budget_ratio) * 3)
        
        # ff budget is critically low, force low fidelity
        if budget_ratio < 0.05:
            selected_point_idx = torch.argmax(mean[:, 0] - torch.sqrt(var[:, 0]))  # Simple LCB for low fidelity
            selected_fidelity = 0
        # in early phase, favor high fidelity to build better model
        elif early_phase and budget_ratio > 0.6:
            # during early phase, use high fidelity more often to build better model if budget allows
            # find most promising point with high fidelity
            selected_point_idx = torch.argmin(mean[:, 1] - 0.5 * torch.sqrt(var[:, 1]))
            selected_fidelity = 1
        # otherwise use our improved acquisition function
        else:
            selected_point_idx, selected_fidelity = efficient_cost_aware_acquisition(
                mean, var, current_best, i - nsamp, niter - nsamp, 
                cost_ratio=effective_cost_ratio
            )
        
        if selected_fidelity == 0:  
            selected_point = low_fidelity_indices[selected_point_idx].unsqueeze(0).item()
            low_fid_count += 1
            spent_budget += 1.0  # low fidelity cost
        else:  
            selected_point = high_fidelity_indices[selected_point_idx].unsqueeze(0).item()
            high_fid_count += 1
            spent_budget += cost_ratio  # high fidelity cost
        
        get_index = torch.cat([get_index, torch.tensor([selected_point])])
        
        # Log the selection
        current_value = y[selected_point].item()
        log.append([selected_point_idx, selected_fidelity, current_value, current_best, spent_budget])
        
        print(f"Iter {i}, Point {selected_point}, Fidelity {selected_fidelity}, "
              f"Value {current_value:.4f}, Best {current_best:.4f}, "
              f"Budget {spent_budget:.1f}/{total_budget:.1f} ({100*budget_ratio:.1f}%)")
    
    return get_index, log, {'low': low_fid_count, 'high': high_fid_count, 'budget': spent_budget}



results = {
    "indices": [],
    'fidelities': [],
    'time': [],
    'log': [],
    'budget_stats': [],
    'convergence': []
}

for i in range(nruns):
    start_time = time.time()
    get_index, log, fidelity_stats = rapid_convergence_mfal_iteration(X_mogp, y_mogp, nsamp, niter, cost_ratio=5.0)
    end_time = time.time()
    
    convergence_history = [entry[3] for entry in log]  # Extract best values from log
    
    print(f'Run: {i}, Time: {end_time-start_time:.2f}s, '
          f'Low fidelity: {fidelity_stats["low"]}, High fidelity: {fidelity_stats["high"]}, '
          f'Budget used: {fidelity_stats["budget"]:.1f}, '
          f'Final best value: {convergence_history[-1]:.6f}')
    
    results["indices"].append(get_index)
    results["fidelities"].append(fidelity_stats)
    results["log"].append(log)
    results['time'].append(end_time-start_time)
    results['convergence'].append(convergence_history)

with open('MFAL_rapid_convergence.pkl', 'wb') as f:
    pickle.dump(results, f)


def plot_convergence(results):
    plt.figure(figsize=(10, 6))
    
    for i, conv in enumerate(results['convergence']):
        plt.plot(conv, alpha=0.3, label=f'Run {i}' if i == 0 else None)
    
    max_len = max(len(conv) for conv in results['convergence'])
    avg_conv = []
    for i in range(max_len):
        values = [conv[i] if i < len(conv) else conv[-1] for conv in results['convergence']]
        avg_conv.append(sum(values) / len(values))
    
    plt.plot(avg_conv, 'r-', linewidth=2, label='Average')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Found Value (Minimum)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mfal_convergence_.png', dpi=300)
    plt.close()

# call this after running experiments
plot_convergence(results)



