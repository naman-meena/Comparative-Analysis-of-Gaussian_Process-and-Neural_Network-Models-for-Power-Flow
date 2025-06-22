# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:05:32 2025

@author: naman
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore")
import torch
import gpytorch
import numpy as np
import pandas as pd
import pandapower as pp
from joblib import Parallel, delayed # used for parallel processing, delayed -->  wraps a function so it can be called in parallel
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # to check whether tf is identifying our gpu, if not we'll need to update cuda, cuddn

######## Sampling our data #######

n_buses = 33 # here 32 load buses, 1 slack bus

def sample_cases(seed=None): # for reproducibility of random numbers
    np.random.seed(seed) # same seed, random numbers
    
    net_data = pp.networks.case33bw()# call imports all the network data (buses, lines, loads, generators, etc.) for distribution grid
    # for each situation
    
    for idx in net_data.load.index: ## scale each load (±20%)
        net_data.load.at[idx, 'p_mw'] *= np.random.uniform(0.8, 1.2) # 32 P values
        net_data.load.at[idx, 'q_mvar'] *= np.random.uniform(0.8, 1.2) # 32 Q values

    pp.runpp(net_data) ## runs a powerflow calculation
    # Storing the features and results
    features = np.hstack([net_data.load['p_mw'].values, net_data.load['q_mvar'].values ]) # input features (loads)
    targets = net_data.res_bus['vm_pu'].values # output voltage magnitudes
    #targets_theta = net_data.res_bus['va_degree'].values # output voltage_angles(theta)
    return features, targets
    
n_cases = 1000



########### PolyKernel Gaussian Process model (define) ##########

class BatchGpModel(gpytorch.models.ExactGP): # one per bus , custom class inheriting from parent class
    def __init__(self, train_inputs, train_targets, likelihood, kernel_type = 'rbf', kernel_parameters = None):
        super() .__init__(train_inputs, train_targets, likelihood)  # parent class calling instructure method for a new instance
        
        batch_shape = torch.Size([train_targets.shape[0]]) #for no. of outputs of n_buses, GPs to run in parallel (one for each bus)
        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape) # constant mean function for each bus (output)


        if kernel_type == 'rbf': # radial basis function
            base_kernel = gpytorch.kernels.RBFKernel( # it is a sqaured eponential kernel
                batch_shape=batch_shape, # want each bus to get its own kernel
                lengthscale_constraint=gpytorch.constraints.Interval( 
                    kernel_parameters.get('lengthscale_min', 0.1),     # low bound for lengthscale                                 
                    kernel_parameters.get('lengthscale_max', 10.0)  # upper bound for lengthscale
                    )
                )
            
        elif kernel_type == 'polynomial': 
            base_kernel = gpytorch.kernels.PolynomialKernel(
                power = kernel_parameters.get('degree', 2), # quadratic
                batch_shape=batch_shape
                )
            
        elif kernel_type == 'linear':
            base_kernel = gpytorch.kernels.LinearKernel(
                batch_shape=batch_shape
                )
            
        else: raise ValueError(f"Unknown Kernel type: {kernel_type}")
        
        self.covariance_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)        # degree to which our model can learn
        
    def forward(self, inputs):  # self --> refering to current isntance of the class
        mean_inputs = self.mean_module(inputs)
        covariance_inputs = self.covariance_module(inputs)
        return gpytorch.distributions.MultivariateNormal(mean_inputs, covariance_inputs) # GP's mean and covariance for the input
    
################## Trainig our model ##########

def train_gpm(model, likelihood, train_inputs, train_targets, training_iterations = 500, lr = 0.1):
    model.train() # training mode
    likelihood.train()  # training mode

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    # will update all parameters of your GP model during training, lr --> how big each update step is
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # the loss function for GP, marginal log likelihood

    for i in range(training_iterations):
        optimizer.zero_grad() ## clears (zeros out) all the gradients from the previous iteration
        output = model(train_inputs) #forward pass
        loss = -mll(output, train_targets).mean() # the loss to be minimized, output --> model's predictions, loss to be minimizzed so thats why a negative sign
       
        loss.backward()  # gradients of the loss with respect to all model parameters
        optimizer.step()   # here step updating of param happening
        
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}")   # converts the tensor loss(0d tensor) to a standard Python float with formatted with 3 decimal places
            
    return model, likelihood
'''
 ##     some theory abbout mll
# the decreasing and increasingly negative loss in your output is a sign that your GP model is learning and fitting the data well        
# they indicate a higher probability fit to the data, maximizing the marginal log likelihood is the standard training objective
'''

############# Testing the model##########

def test_model(model, likelihood, test_inputs, test_targets):
    model.eval()
    likelihood.eval() 
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():  # disables gradient calculations
        predictions = likelihood(model(test_inputs))   # test inputs through the model and likelihood
        mean_pred = predictions.mean.t().cpu().numpy()  # transposes them (so rows = scenarios, columns = buses), moves them to CPU and converts to a Numpy array
        
    actual = test_targets.t().cpu().numpy() # transposes target tensor, moves it to CPU and converts it to a numpy array
    
    # now both mean_pred and actual have shape (n_cases, n_buses)
    
    metrics = {} # empty dict to story ( rmse , mse , mae)
    
    '''looping over each bus for metrics'''
    for bus_id in  range(actual.shape[1]):
        actual_bus = actual[:, bus_id]
        predicted_bus = mean_pred[:, bus_id]
        
        
        mse = mean_squared_error(actual_bus, predicted_bus)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_bus, predicted_bus)
        
        metrics[f'bus_{bus_id+1}'] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae} # stores the metrics in the dictionary, using the bus number as the key
    return metrics
        
############ Optuna to optimize hyperparameters for gp ##########

def objective(trial, kernel_type, X_train, Y_train, X_test, Y_test, n_buses): # function for Optuna's optimization, this func will be called for each trial (for hyperparam)
    if kernel_type == 'rbf':
        kernel_parameters = { # suggest_float---> samples a float within the given range
            'lengthscale_min' : trial.suggest_float('lengthscale_min', 0.01, 1.0), # suggests hyperparameters for the kernel
            'lengthscale_max' : trial.suggest_float('lengthscale_max', 1.0, 20.0)
            }
        lr = trial.suggest_float('lr', 0.01, 0.3) #lr-> learnig rate using suggest_float
            
    elif kernel_type == 'polynomial':
        kernel_parameters = {
            'degree': trial.suggest_int('degree', 2, 4) # suggests hyperparameters for the kernel
            }
        lr = trial.suggest_float('lr', 0.01, 0.2)
        
    elif kernel_type == 'linear':
        kernel_parameters = {   }
        lr = trial.suggest_float('lr', 0.005, 0.1)
        
    training_iterations = trial.suggest_int('training_iterations', 300, 800) # number of training epochs
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_buses])).to(device) # how well the GP model fits the observed data
    #shifted to gpu
    model = BatchGpModel(X_train, Y_train, likelihood, kernel_type, kernel_parameters).to(device) #shifted to gpu
    
    model, likelihood = train_gpm(model, likelihood, X_train, Y_train, training_iterations, lr) ## trains teh MODEL
    
    metrics = test_model(model, likelihood, X_test, Y_test) ## tests on test data
    
    avg_rmse = np.mean([metrics[f'bus_{i+1}']['RMSE'] for i in range(n_buses)])
    
    return avg_rmse

######### To find the best hyperparameters for a specific kernel #######
def run_kernel_optimization(kernel_type: str, X_train, Y_train, X_test, Y_test, n_buses: int, n_trials: int = 35): #goal is to optimize the GP model's hyperparameters , n_trials default set to 35
    print(f"\n=== Optimizing {kernel_type.upper()} Kernel ===") ## which kernel is being optimized
    
    study = optuna.create_study(direction='minimize') # optuna study that aims to minimize the objective function
    study.optimize(
        lambda trial: objective(trial, kernel_type, X_train, Y_train, X_test, Y_test, n_buses), n_trials=n_trials)

    
    print(f"Best {kernel_type} parameters: {study.best_params}")
    print(f"Best {kernel_type} RMSE: {study.best_value:.6f}")
    
    if kernel_type == 'rbf':
        kernel_parameters = {k: v for k, v in study.best_params.items() if k.startswith('lengthscale')}
    elif kernel_type == 'polynomial':
        kernel_parameters = {'degree': study.best_params.get('degree', 2)}
    else:
        kernel_parameters = {}
    # extracts the relevant hyperparameters from best result (after optimisaton)
    
    final_likelihood = gpytorch.likelihoods.GaussianLikelihood( batch_shape= torch.Size([n_buses])).to(device) # batch shape matching the number of output buses
    
    final_model = BatchGpModel(X_train, Y_train, final_likelihood, kernel_type, kernel_parameters).to(device) # moving to gpu
    
    final_model, final_likelihood = train_gpm(  # trains the final model using the best number of training iterations and learning rate found
        final_model, final_likelihood, X_train, Y_train,
        study.best_params['training_iterations'],
        study.best_params['lr']
    )
    
    final_metrics = test_model(final_model, final_likelihood, X_test, Y_test) #  computing metrics
    
    return study, final_metrics


### Result Tables ######

def create_results_tables(results_dict: Dict, max_buses: int = 33):
    tables = {}
    
    for kernel_name, metrics in results_dict.items():  # iterates over each kernel in the results dict
        bus_range = min(len(metrics), max_buses)
        
        data = []
        for bus_id in range(bus_range):
            bus_metrics = metrics[f'bus_{bus_id+1}']
            data.append({
                'Load Bus': bus_id + 1,
                'RMSE': bus_metrics['RMSE'],
                'MSE': bus_metrics['MSE'],
                'MAE': bus_metrics['MAE']
            })
        
        df = pd.DataFrame(data)  ## converts the list of dict into a pandas dataframe
        
        summary_stats = {
            'Metric': ['RMSE', 'MSE', 'MAE'],
            'Min': [df['RMSE'].min(), df['MSE'].min(), df['MAE'].min()],
            'Max': [df['RMSE'].max(), df['MSE'].max(), df['MAE'].max()]
        }
        
        summary_df = pd.DataFrame(summary_stats) # same asdata converted earlier
        
        tables[kernel_name] = {
            'detailed': df,
            'summary': summary_df
        }
    
    return tables

##### Execcuting (giving valuies or inputs) #########

print("Generating scenarios...")
n_scenarios = n_cases
results = Parallel(n_jobs=-1)(delayed(sample_cases)(i) for i in range(n_scenarios)) # uses all available CPU cores

Xp, Yp = zip(*results)  # results is a list of (features, targets) tuples
                        # Xp--> all feature vectors (one per scenario)
                        # Yp--> all target vectors (one per scenario)

Xp = np.array(Xp)  # from tuples to numpy arrays
Yp = np.array(Yp)


### NumPy arryas to PyTorch tensors for gpu processing ####

Xp_tensor = torch.tensor(Xp, dtype=torch.float32, device=device)   #same size as earlier
Yp_tensor = torch.tensor(Yp, dtype=torch.float32, device=device)

n_scenarios, n_features = Xp_tensor.shape
n_buses = Yp_tensor.shape[1] #in Yp .shape[1] will give the bus column , .shape[0] will of nscenarios

Xp_batched = Xp_tensor.unsqueeze(0).repeat(n_buses, 1, 1) # (n_buses, n_scenarios, n_features), new dimension to Xp_tensor and repeats it for each bus
Yp_batched = Yp_tensor.t()

train_size = int(0.8 * n_scenarios) # training samples as 80% of the total scenarios
X_train = Xp_batched[:, :train_size, :]
Y_train = Yp_batched[:, :train_size]
X_test = Xp_batched[:, train_size:, :]
Y_test = Yp_batched[:, train_size:]

kernels = ['rbf', 'polynomial', 'linear'] # kernel types to be used

all_results = {} # empty dict initiallised

for kernel in kernels:
    study, metrics = run_kernel_optimization(kernel, X_train, Y_train, X_test, Y_test, n_buses, n_trials=30) # n_trials explicitly passed
    
    all_results[kernel] = metrics

results_tables = create_results_tables(all_results, max_buses=32)

for kernel_name, tables in results_tables.items():
    print(f"\n" + "="*50)
    print(f"{kernel_name.upper()} KERNEL RESULTS")
    print("="*50)
    
    print(f"\nSummary Statistics (Min and Max across buses 1-33):")
    print(tables['summary'].to_string(index=False, float_format='%.6f'))  # upto 6 decimal float points
    
    print(f"\nDetailed Results for first 10 buses:")
    print(tables['detailed'].head(10).to_string(index=False, float_format='%.6f')) # same
    
    tables['detailed'].to_csv(f'{kernel_name}_detailed_results.csv', index=False)
    tables['summary'].to_csv(f'{kernel_name}_summary_results.csv', index=False)

print(f"\nAll results saved!!")
