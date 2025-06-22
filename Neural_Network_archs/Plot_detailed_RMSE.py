import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load each CSV and add a column for the kernel type
files_kernels = [
    ('NN3_2Layers_Sigmoid_per_bus_metrics.csv', 'NN3_2Layers_Sigmoid'),
    ('NN4_3Layers_Sigmoid_per_bus_metrics.csv', 'NN4_3Layers_Sigmoid'),
    ('NN1_2Layers_ReLU_per_bus_metrics.csv', 'NN1_2Layers_ReLU'),
    ('NN2_3Layers_ReLU_per_bus_metrics.csv', 'NN2_3Layers_ReLU')
]

dfs = []
for file, kernel in files_kernels:
    df = pd.read_csv(file)
    df['Kernel'] = kernel
    dfs.append(df)

# Combine all data
all_data = pd.concat(dfs, ignore_index=True)

plt.figure(figsize=(12, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
linestyles = ['-', '-', '-', '-']  # All solid lines

# Plot RMSE for each kernel
for i, kernel in enumerate(['NN1_2Layers_ReLU', 'NN2_3Layers_ReLU', 
                           'NN3_2Layers_Sigmoid', 'NN4_3Layers_Sigmoid']):
    kernel_data = all_data[all_data['Kernel'] == kernel].sort_values('Bus')
    plt.plot(kernel_data['Bus'], kernel_data['RMSE'], 
             color=colors[i], marker=markers[i], markersize=6, 
             linewidth=2, label=kernel, alpha=0.8, linestyle=linestyles[i])

plt.xlabel('Load Bus', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Detailed RMSE Across Load Buses for Different Neural Network Architectures', 
          fontsize=14, fontweight='bold')

# Add grid
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

plt.xlim(0, 33)
plt.ylim(0, None)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout()

plt.show()
