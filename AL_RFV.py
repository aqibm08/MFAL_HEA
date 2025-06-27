import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time
from scipy.stats import norm
from sklearn.decomposition import PCA
from torch.quasirandom import SobolEngine
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

filterwarnings('ignore')

def load_and_prepare_data(data_path):
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    
    X = data[['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE','Density','TermalC','MP','FI','SI','TI','M']]
    y = data[['TEC']]
    
    pos_mask = y.values.flatten() >= 0
    X = X.iloc[pos_mask]
    y = y.iloc[pos_mask]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=5)
    X_transformed = pca.fit_transform(X_scaled)
    
    X_tensor = torch.from_numpy(X_transformed).to(torch.float64)
    y = np.reshape(y, (np.size(y), 1))
    y_tensor = torch.from_numpy(y).to(torch.float64)
    
    return X, y, X_tensor, y_tensor, X_transformed, scaler, pca

def generate_exploration_point(X_bounds):
    """Generate a random point within the bounds of the data for exploration."""
    dim = X_bounds.shape[1]
    sobol = SobolEngine(dimension=dim, scramble=True)
    random_point = sobol.draw(1).to(torch.float64)
    
    for d in range(dim):
        random_point[:, d] = random_point[:, d] * (X_bounds[1, d] - X_bounds[0, d]) + X_bounds[0, d]
    return random_point

def expected_improvement_min(mean, std, best_f, epsilon=0.01):
    """EI acquisition function for minimisation."""
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).to(torch.float64)
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).to(torch.float64)
    if isinstance(best_f, (float, int)):
        best_f = torch.tensor(best_f, dtype=torch.float64)
        
    with torch.no_grad():
        # for minimization, improvement happens when mean < best_f
        improvement = best_f - mean - epsilon
        
        # calculate EI using the formula
        z = improvement / (std + 1e-9)  # small constant to avoid division by zero
        cdf = torch.tensor(norm.cdf(z.numpy()))
        pdf = torch.tensor(norm.pdf(z.numpy()))
        
        ei = improvement * cdf + std * pdf
        # we want to select points with HIGHEST EI values
        return ei

def lcb(mean, std, beta, iterations_without_improvement):
    """LCB acquisition function for minimisation."""
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).to(torch.float64)
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).to(torch.float64)
        
    #we want points with lowest (mean - beta*std)
    adjusted_beta = max(0.5, beta / (1 + 0.1 * iterations_without_improvement))
    return mean - adjusted_beta * std

def beta_schedule(iteration):
    """dynamic schedule for beta parameter in LCB."""
    # increased exploration factor for LCB
    return max(0.5, 3.0 * (1 - iteration / 500))

def hyperparam_rf(X, y):
    """random forest variance hyperparameter tuning"""
    params = {
        'n_estimators': [50, 100, 200, 300, 400, 500],  
        'max_depth': [5, 10, 15, 20, 25, 30],        
        'min_samples_split': [1, 2, 3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
        'bootstrap': [True]
    }
    model = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,  
        cv=3,      
        verbose=0,
        random_state=42,
        n_jobs=-1  
    )
    # y is flattened properly for the fit
    y_flat = y.ravel() if hasattr(y, 'ravel') else y.squeeze()
    rf_search.fit(X, y_flat)
    best_params = rf_search.best_params_
    print(f"Best hyperparameters: {best_params}")
    return best_params

def al_hea_rf(X, y, acq, niter, nsamp, exploration_prob=0.1, beta_schedule=None, verbose=False):
    """
    Active learning function with Random Forest surrogate model for direct minimization.
    
    Args:
        X: Input features tensor or numpy array
        y: Target values tensor or numpy array to minimize
        acq: Acquisition function name ('EI' or 'LCB')
        niter: Number of iterations
        nsamp: Initial sample size
        exploration_prob: Probability of random exploration
        beta_schedule: Function that returns beta value based on iteration
        verbose: Whether to print detailed progress
    """
    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
        
    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = y
        
    X_bounds = np.vstack([np.min(X_np, axis=0), np.max(X_np, axis=0)])
    X_bounds = torch.tensor(X_bounds)
    # random samples to initialize    
    get_index = np.random.choice(np.arange(len(y_np)), size=nsamp, replace=False)
    
    best_value_so_far = float('inf')  # for minimization
    iterations_without_improvement = 0
    
    metrics = {
        'iteration': [],
        'selected_point': [],
        'selected_value': [],
        'best_value_so_far': [],
        'hyperparams': [],
        'acquisition_values': [],
        'exploration_flag': []
    }
    
    for j in range(nsamp):
        metrics['iteration'].append(j)
        metrics['selected_point'].append(get_index[j])
        metrics['selected_value'].append(float(y_np[get_index[j]][0]))
        current_best = float(y_np[get_index[:j+1]].min())
        metrics['best_value_so_far'].append(current_best)
        metrics['hyperparams'].append(None)
        metrics['acquisition_values'].append(None)
        metrics['exploration_flag'].append(False)
        
        if current_best < best_value_so_far:
            best_value_so_far = current_best
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
    
    # main loop
    for i in range(nsamp, niter):
        current_exploration_prob = exploration_prob * (1 + 0.1 * iterations_without_improvement)
        
        # decide whether to explore or exploit
        do_exploration = np.random.random() < current_exploration_prob
        
        X_train = X_np[get_index]
        y_train = y_np[get_index]
        
        if verbose:
            current_min = float(y_train.min())
            current_max = float(y_train.max())
            print(f"Iter {i}: Training data range - Min: {current_min}, Max: {current_max}")
        
        # fit Random Forest model
        try:
            # Get best hyperparameters for the RF model
            if i == nsamp or i % 10 == 0:  # update hyperparameters periodically
                best_params = hyperparam_rf(X_train, y_train)
            else:
                # use the last set of hyperparameters to save computation
                best_params = metrics['hyperparams'][-1] if metrics['hyperparams'][-1] is not None else {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'bootstrap': True
                }
                
            # train the RF model with best hyperparameters
            rf_model = RandomForestRegressor(**best_params, random_state=42)
            y_train_flat = y_train.ravel() if hasattr(y_train, 'ravel') else y_train.squeeze()
            rf_model.fit(X_train, y_train_flat)
            
            #get predictions and uncertainty from all trees
            preds = np.array([tree.predict(X_np) for tree in rf_model.estimators_])
            mean = np.mean(preds, axis=0)
            var = np.var(preds, axis=0)
            std = np.sqrt(var)
            
            # set beta for LCB based on iteration if schedule provided
            if beta_schedule and acq == 'LCB':
                beta = beta_schedule(i)
            else:
                beta = 2.0  # default
                
            # calculate acquisition values based on method
            if acq == 'EI':
                best_f = float(y_train.min())  # best observed value (minimum for minimization)
                acquisition_value = expected_improvement_min(mean, std, best_f)
                
                # for EI, we want to select points with HIGHEST EI values
                masked_acq = acquisition_value.clone()
                masked_acq[get_index] = float('-inf')  
                
                # check if there are any positive EI values
                if torch.max(masked_acq) <= 0:
                    do_exploration = True
                else:
                    index_max = torch.argmax(masked_acq).item()
                    
            elif acq == 'LCB':
                acquisition_value = lcb(mean, std, beta, iterations_without_improvement)
                
                # for LCB, we want to select points with LOWEST LCB values
                masked_acq = acquisition_value.clone()
                masked_acq[get_index] = float('inf')  
                index_max = torch.argmin(masked_acq).item()
            else:
                raise ValueError(f"Unknown acquisition function: {acq}")
                
            # select next point
            if do_exploration:
                exploration_point = generate_exploration_point(X_bounds)
                
                with torch.no_grad():
                    distances = torch.cdist(exploration_point, torch.tensor(X_np, dtype=torch.float64))
                    index_max = torch.argmin(distances).item()
                    
        except Exception as e:
            if verbose:
                print(f"Model fitting failed at iteration {i}: {e}")
            # fall back to random selection if model fitting fails
            remaining_indices = np.setdiff1d(np.arange(len(y_np)), get_index)
            if len(remaining_indices) > 0:
                index_max = np.random.choice(remaining_indices)
            else:
                index_max = np.random.choice(np.arange(len(y_np)))
            do_exploration = True
            best_params = None
            
        # add selected point to our dataset
        get_index = np.append(get_index, index_max)
        
        # update best value and stagnation counter
        current_min = float(y_np[get_index].min())
        if current_min < best_value_so_far:
            best_value_so_far = current_min
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
            
        if verbose:
            print(f"Iter {i}, Selected point: {index_max}, Value: {y_np[index_max][0]}, Best: {current_min}, " + 
                  f"{'EXPLORE' if do_exploration else 'EXPLOIT'}, Stagnant: {iterations_without_improvement}")
        
        # store metrics
        metrics['iteration'].append(i)
        metrics['selected_point'].append(index_max)
        metrics['selected_value'].append(float(y_np[index_max][0]))
        metrics['best_value_so_far'].append(current_min)
        metrics['hyperparams'].append(best_params)
        metrics['exploration_flag'].append(do_exploration)
        
        try:
            if acq == 'EI' or acq == 'LCB':
                acq_info = {
                    'selected_value': float(acquisition_value[index_max].item()) if not do_exploration else None,
                    'statistics': {
                        'mean': float(acquisition_value.mean().item()) if not do_exploration else None,
                        'std': float(acquisition_value.std().item()) if not do_exploration else None,
                        'min': float(acquisition_value.min().item()) if not do_exploration else None,
                        'max': float(acquisition_value.max().item()) if not do_exploration else None
                    }
                }
        except:
            acq_info = None
            
        metrics['acquisition_values'].append(acq_info)
    
    return get_index, metrics
def additional_metrics(indices, X, y, true_min):
    """calculate additional performance metrics for minimization."""
    if true_min is None:
        true_min = np.min(y)
    
    best_values = [np.min(y[indices[:i+1]]) for i in range(len(indices))]
    simple_regret = [float(val - true_min) for val in best_values]
    
    diversity = []
    for i in range(len(indices)):
        if i > 0:
            selected_points = X[indices[:i+1]]
            dists = []
            for j in range(len(selected_points)):
                for k in range(j+1, len(selected_points)):
                    dists.append(float(np.linalg.norm(selected_points[j] - selected_points[k])))
            diversity.append(float(np.mean(dists) if dists else 0))
        else:
            diversity.append(0.0)
    
    # convergence metrics
    iterations_to_110pct = next((i for i, val in enumerate(best_values) if val <= 1.1 * true_min), len(indices))
    iterations_to_105pct = next((i for i, val in enumerate(best_values) if val <= 1.05 * true_min), len(indices))
    iterations_to_101pct = next((i for i, val in enumerate(best_values) if val <= 1.01 * true_min), len(indices))
    
    return {
        'best_values': best_values,
        'simple_regret': simple_regret,
        'cumulative_regret': np.cumsum(simple_regret).tolist(),
        'spatial_diversity': diversity,
        'iterations_to_110pct': iterations_to_110pct,
        'iterations_to_105pct': iterations_to_105pct,
        'iterations_to_101pct': iterations_to_101pct
    }

#def plot_optimization_progress(best_values, true_min, acq, run_id=None):
#    """Plot optimization progress."""
#    plt.figure(figsize=(12, 6))
#    plt.plot(best_values, label=f'Best CTE value ({acq})')
#    plt.axhline(y=true_min, color='r', linestyle='--', label=f'True minimum: {true_min:.4f}')
#    plt.xlabel('Iteration')
#    plt.ylabel('Best TEC value found')
#    
#    title = f'Optimization Progress with {acq}'
#    if run_id is not None:
#        title += f' - Run {run_id}'
#    plt.title(title)
#    
#    plt.legend()
#    plt.grid(True)
#    
#    filename = f'RFV_optimization_progress_{acq}'
#    if run_id is not None:
#        filename += f'_run{run_id}'
#    plt.savefig(f'{filename}.png')
#    plt.close()
#def plot_acquisition_comparison(results, true_min):
#    """Plot comparison of acquisition functions."""
#    plt.figure(figsize=(12, 6))
#    colors = ['b', 'g']
#    for i, acq in enumerate(results.keys()):
#        # Calculate average best value across runs
#        all_best_values = np.array([run['performance_metrics']['best_values'] for run in results[acq]['run_results']])
#        avg_best_values = np.mean(all_best_values, axis=0)
#        
#        # Plot with confidence intervals
#        std_best_values = np.std(all_best_values, axis=0)
#        plt.plot(avg_best_values, color=colors[i], label=f'{acq}')
#        plt.fill_between(
#            range(len(avg_best_values)),
#            avg_best_values - std_best_values,
#            avg_best_values + std_best_values,
#            color=colors[i], alpha=0.2
#        )
#    
#    plt.axhline(y=true_min, color='r', linestyle='--', label=f'True minimum: {true_min:.4f}')
#    plt.xlabel('Iteration')
#    plt.ylabel('Best TEC value found')
#    plt.title('Comparing Acquisition Methods for Minimization')
#    plt.legend()
#    plt.grid(True)
#    plt.savefig('acquisition_comparison.png')
#    plt.close()

def print_optimal_points(results, X, y, X_transformed, scaler, pca):
    """Print information about the optimal points found."""
    print("\n===== OPTIMAL POINTS FOUND =====")
    for acq in results.keys():
        # find the best run (one with lowest final value)
        best_run_idx = np.argmin([run['performance_metrics']['best_values'][-1] 
                                 for run in results[acq]['run_results']])
        best_run = results[acq]['run_results'][best_run_idx]
        best_indices = best_run['indices']
        best_values = [float(y[idx][0]) for idx in best_indices]
        best_idx = best_indices[np.argmin(best_values)]
        
        # get the original (pre-PCA) features for the optimal point
        optimal_point_pca = X_transformed[best_idx]
        
        # reshape to ensure optimal_point_pca is 2D before inverse transform
        optimal_point_pca_reshaped = optimal_point_pca.reshape(1, -1)
        
        # transform back to original feature space (approximate)
        optimal_point_scaled = pca.inverse_transform(optimal_point_pca_reshaped)
        optimal_point = scaler.inverse_transform(optimal_point_scaled)
        
        print(f"\nBest point found with {acq} (Run {best_run_idx+1}):")
        print(f"TEC value: {y[best_idx][0]}")
        print("Feature values:")
        for feature, value in zip(X.columns, optimal_point[0]):  # Now accessing first row of optimal_point
            print(f"  {feature}: {value:.4f}")
            
        top_indices = best_indices[np.argsort([y[idx][0] for idx in best_indices])[:5]]
        print(f"\nTop 5 MINIMUM points found with {acq} (verifying minimization):")
        for idx in top_indices:
            print(f"  Point index: {idx}, TEC value: {y[idx][0]}")

def print_performance_comparison(results, true_min):
    """Print performance comparison between acquisition functions."""
    print("\n===== PERFORMANCE COMPARISON =====")
    for acq in results.keys():
        metrics = results[acq]['aggregated_metrics']
        print(f"\n{acq} Acquisition Function:")
        print(f"  Best CTE value found: {metrics['best_value_found']:.6f}")
        print(f"  True minimum TEC value: {metrics['true_min']:.6f}")
        print(f"  Gap to true minimum: {(metrics['best_value_found'] - metrics['true_min']):.6f}")
        print(f"  Iterations to within 10% of minimum: {metrics['avg_iterations_to_110pct']:.1f}")
        print(f"  Iterations to within 5% of minimum: {metrics['avg_iterations_to_105pct']:.1f}")
        print(f"  Iterations to within 1% of minimum: {metrics['avg_iterations_to_101pct']:.1f}")
        print(f"  Total runtime: {metrics['time']:.2f} seconds")

#def plot_regret(best_values, true_min, simple_regret, acq, run_id=None):
#    """Plot optimization regret."""
#    plt.figure(figsize=(14, 8))
#    
#    plt.subplot(2, 1, 1)
#    plt.plot(best_values, label=f'Best TEC value ({acq})')
#    plt.axhline(y=true_min, color='r', linestyle='--', label=f'True minimum: {true_min:.4f}')
#    plt.xlabel('Iteration')
#    plt.ylabel('Best TEC value found')
#    plt.title(f'Optimization Progress with {acq}')
#    plt.legend()
#    plt.grid(True)
#    
#    plt.subplot(2, 1, 2)
#    plt.plot(simple_regret, label='Simple Regret')
#    plt.yscale('log')  # Log scale to better see the convergence
#    plt.xlabel('Iteration')
#    plt.ylabel('Regret (log scale)')
#    plt.title('Optimization Regret')
#    plt.grid(True)
#    plt.tight_layout()
#    
#    filename = f'RFV_optimization_regret_{acq}'
#    if run_id is not None:
#        filename += f'_run{run_id}'
#    plt.savefig(f'{filename}.png')
#    plt.close()
def plot_exploration_exploitation(iterations, selected_values, exploration_flags, best_values, acq, run_id=None):
    """Plot exploration vs exploitation."""
    plt.figure(figsize=(12, 6))
    
    plt.scatter([i for i, flag in zip(iterations, exploration_flags) if not flag], 
                [v for v, flag in zip(selected_values, exploration_flags) if not flag],
                c='blue', label='Exploitation points', alpha=0.7)
    plt.scatter([i for i, flag in zip(iterations, exploration_flags) if flag], 
                [v for v, flag in zip(selected_values, exploration_flags) if flag],
                c='red', label='Exploration points', alpha=0.7)
    
    plt.plot(iterations, best_values, 'g-', label='Best value so far')
    
    plt.xlabel('Iteration')
    plt.ylabel('TEC Value')
    plt.title(f'Exploration vs Exploitation ({acq})')
    plt.legend()
    plt.grid(True)
    
    filename = f'RFV_exploration_exploitation_{acq}'
    if run_id is not None:
        filename += f'_run{run_id}'
    plt.savefig(f'{filename}.png')
    plt.close()


def main():
    # Configuration
    acquisition_methods = ['EI', 'LCB']
    nruns = 100  # 100 runs
    niter = 500
    nsamp = 20
    data_path = 'Data_base.csv'
    
    sns.set_style('whitegrid')
    
    # load and prepare data
    print("Loading and preparing data...")
    X, y, X_tensor, y_tensor, X_transformed, scaler, pca = load_and_prepare_data(data_path)
    
    print("Shape of X:", np.shape(X_transformed))
    print("Shape of y:", np.shape(y))
    print("Minimum TEC value:", y.min())
    print("Maximum TEC value:", y.max())
    
    # true minimum for reference
    true_min = float(np.min(y))
    print(f"True minimum TEC value: {true_min}")
    
    results = {}
    
    for acq in acquisition_methods:
        print(f"\n==== Running Active Learning with {acq} for {nruns} runs ====")
        run_results = []
        start_time = time.time()
        
        if acq == 'LCB':
            exploration_prob = 0.05  # higher exploration for LCB
        else:
            exploration_prob = 0.1   # lower for EI which has inherent exploration
        
        for run in tqdm(range(nruns), desc=f"{acq} runs"):
            indices, iteration_metrics = al_hea_rf(
                X_tensor, 
                y_tensor, 
                acq, 
                niter, 
                nsamp, 
                exploration_prob=exploration_prob,
                beta_schedule=None,
                verbose=(run == 0)  
            )
            
            performance_metrics = additional_metrics(indices, X_transformed, y, true_min)
            run_results.append({
                'indices': indices,
                'iteration_metrics': iteration_metrics,
                'performance_metrics': performance_metrics
            })
            
            # Plot results for the first run only to avoid too many plots
            if run == 0:
                best_values = performance_metrics['best_values']
                simple_regret = performance_metrics['simple_regret']
                #plot_optimization_progress(best_values, true_min, acq, run+1)
                #plot_regret(best_values, true_min, simple_regret, acq, run+1)
                
                iterations = iteration_metrics['iteration']
                selected_values = iteration_metrics['selected_value']
                exploration_flags = iteration_metrics['exploration_flag']
                best_values_over_time = []
                current_best = float('inf')
                for val in selected_values:
                    current_best = min(current_best, val)
                    best_values_over_time.append(current_best)
                
                plot_exploration_exploitation(iterations, selected_values, exploration_flags, 
                                             best_values_over_time, acq, run+1)
        
        end_time = time.time()
        time_acq = end_time - start_time
        print(f"Total time for {acq}: {time_acq:.2f} seconds")
        
        best_values_found = [min(run['performance_metrics']['best_values']) for run in run_results]
        aggregated_metrics = {
            'time': time_acq,
            'best_value_found': min(best_values_found),
            'avg_best_value': np.mean(best_values_found),
            'std_best_value': np.std(best_values_found),
            'avg_iterations_to_110pct': np.mean([r['performance_metrics']['iterations_to_110pct'] for r in run_results]),
            'avg_iterations_to_105pct': np.mean([r['performance_metrics']['iterations_to_105pct'] for r in run_results]),
            'avg_iterations_to_101pct': np.mean([r['performance_metrics']['iterations_to_101pct'] for r in run_results]),
            'true_min': true_min
        }
        
        results[acq] = {
            'run_results': run_results,
            'aggregated_metrics': aggregated_metrics
        }
    
    print("\nSaving results to pickle file...")
    with open('AL_RF_100runs_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print_optimal_points(results, X, y, X_transformed, scaler, pca)
    print_performance_comparison(results, true_min)
    
#    plot_acquisition_comparison(results, true_min)
    
    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()