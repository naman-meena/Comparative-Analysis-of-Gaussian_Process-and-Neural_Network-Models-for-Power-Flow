# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:24:41 2025

@author: naman
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandapower as pp
from joblib import Parallel, delayed # used for parallel processing, delayed--> wraps a function so tht it can be used in parallel 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # to check whether tf is identifying our gpu, if not we'll need to update cuda, cuddn

## Sampling###

n_buses = 32 # here 32 load buses, 1 slack bus was excluded

def sample_scenario(seed=None):  # for reproducibility of random numbers
    np.random.seed(seed) # same seed, random numbers
    
    net_data = pp.networks.case33bw()#call imports all the network data (buses, lines, loads, generators, etc.) for distribution grid
    # for each situation
    
    for idx in net_data.load.index: ## scale each load (±20%)
        net_data.load.at[idx, 'p_mw'] *= np.random.uniform(0.8, 1.2) # 32 P values
        net_data.load.at[idx, 'q_mvar'] *= np.random.uniform(0.8, 1.2) # 32 Q values
    
    pp.runpp(net_data)
    # Storing the features and results
    features = np.hstack([net_data.load['p_mw'].values, net_data.load['q_mvar'].values]) # input features (loads)
    targets = net_data.res_bus['vm_pu'].values[1:] # output voltage magnitudes, starting from buss 1 instaead of bus 0
    return features, targets

n_scenarios = 1000
results = Parallel(n_jobs=-1)(delayed(sample_scenario)(i) for i in range(n_scenarios)) # uses all available CPU cores

Xp, Yp = zip(*results) # results is a list of (features, targets) tuples
                        # Xp--> all feature vectors (one per scenario)
                        # Yp--> all target vectors (one per scenario)

Xp = np.array(Xp) # from tuples to numpy arrays
Yp = np.array(Yp)

# Xp.shape[0]=no. of scenarios, Xp.shape[1]=64 features

train_size = int(0.8 * n_scenarios)    # splitting the data into 80% traning and 20% testing sets
X_train, X_test = Xp[:train_size], Xp[train_size:] # 800, 64(32 active, 32 reactive)  , 200, 64
Y_train, Y_test = Yp[:train_size], Yp[train_size:] # 800, 32(total buses), 200,32


############### Defining our NN architectures ##############

'''2 layer NN, ReLU'''
def nn1(input_size, output_size):  # keras nneed input_shape as a tuple , (input_size,)---> becomes a tuple with use of comma
    model = keras.Sequential([  # input-->number of features (loads)
        layers.Dense(64, activation = 'relu', input_shape = (input_size,)),  # 64  neurons , reLu--> rectified linear unit outputs the input directly if it above some thrishold other wise zro
        layers.Dropout(0.2), # preventing overfitting , overfitting-->  its noise and random fluctuations are also learned  which is undesirable
        
        layers.Dense(32, activation = 'relu'),  # 32  neurons
        layers.Dense(output_size)# No activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) # mean squared error loss -mse, mean absolute error-mae
    return model

'''3 layer NN, ReLU'''
def nn2(input_size, output_size):
    model = keras.Sequential([ 
        layers.Dense(128, activation = 'relu', input_shape = (input_size,)), #128 neuroons
        layers.Dropout(0.2),
        
        layers.Dense(64, activation = 'relu'),   # 64  neurons
        layers.Dropout(0.2),
        
        layers.Dense(32, activation = 'relu'), # 32  neurons
        layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

'''2 layer NN, sigmmoid'''
def nn3(input_size, output_size): 
    model = keras.Sequential([ 
        layers.Dense(64, activation = 'sigmoid', input_shape = (input_size,)),  # 64  neurons
        layers.Dropout(0.2), 
        
        layers.Dense(32, activation = 'sigmoid'), # 32  neurons
        layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

'''3 layer NN, sigmoid'''
def nn4(input_size, output_size):
    model = keras.Sequential([ 
        layers.Dense(128, activation = 'sigmoid', input_shape = (input_size,)), #128 neuroons
        layers.Dropout(0.2),
        
        layers.Dense(64, activation = 'sigmoid'),   # 64  neurons
        layers.Dropout(0.2),
        
        layers.Dense(32, activation = 'sigmoid'), # 32  neurons
        layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
    return model

###### Training and Testing the Model #########

models = [  # a list of tuples  as (name, model_fucn)
    ('NN1_2Layers_ReLU', nn1),
    ('NN2_3Layers_ReLU', nn2),
    ('NN3_2Layers_Sigmoid', nn3),
    ('NN4_3Layers_Sigmoid', nn4),
]

results_m = []

for name, model_func in models:   # a for loop going through every tuple in models
    print(f"\nTraning {name}....")
    
    model = model_func(X_train.shape[1], Y_train.shape[1])  # input -> features,loads  output->buses
    
    model.fit(X_train, Y_train, epochs=500, validation_split=0.2, verbose=0) # 20% of training data used for validation, can put verbose =1 if want to see progress
    
    predictions = model.predict(X_test)  # predicts bus voltages on test data
    
    mse = mean_squared_error(Y_test, predictions)  # means predicted on array ( sample, outputs)
    mae = mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, predictions) # coefficient of determination (how well predictions match true values)
    
    # Per bus too
    mse_per_bus = np.mean((Y_test - predictions)**2, axis=0)
    mae_per_bus = np.mean(np.abs(Y_test - predictions), axis=0)
    rmse_per_bus = np.sqrt(mse_per_bus) 
    r2_per_bus = r2_score(Y_test, predictions, multioutput='raw_values')  # one R² per bus
    
    results_m.append({
        'name' : name,
        'model' : model,
        'mse' : mse,
        'mae' : mae,
        'rmse' : rmse,
        'mse_per_bus' : mse_per_bus,
        'mae_per_bus' : mae_per_bus,
        'rmse_per_bus' : rmse_per_bus,
        'r2': r2,
        'r2_per_bus': r2_per_bus
    })
    
## Making tables (summarizing)#########

summary = []

for r in results_m:# for loop on results dic
   row = {'Model': r['name']}
   for i in range(n_buses):
        row[f'Min_RMSE_bus_{i+1}'] = r['rmse_per_bus'][i]
        row[f'Max_RMSE_bus_{i+1}'] = r['rmse_per_bus'][i]
        row[f'Min_MSE_bus_{i+1}'] = r['mse_per_bus'][i]
        row[f'Max_MSE_bus_{i+1}'] = r['mse_per_bus'][i]
        row[f'Min_MAE_bus_{i+1}'] = r['mae_per_bus'][i]
        row[f'Max_MAE_bus_{i+1}'] = r['mae_per_bus'][i]
        row[f'Min_R2_bus_{i+1}'] = r['r2_per_bus'][i]
        row[f'Max_R2_bus_{i+1}'] = r['r2_per_bus'][i]
   summary.append(row)
   
    
summary_df = pd.DataFrame(summary)   # maki a dataframe of summary

print("\nSummary Table :")
print(summary_df)

#### Saving result######

for r in results_m:
    pd.DataFrame({
        "Bus": np.arange(1, n_buses+1),
        "MSE": r['mse_per_bus'],
        "RMSE": r['rmse_per_bus'],
        "MAE": r['mae_per_bus'],
        "R2": r['r2_per_bus'], }).to_csv(f"{r['name']}_per_bus_metrics.csv", index=False)
    
summary_df.to_csv('summary_metrics.csv', index=False)
