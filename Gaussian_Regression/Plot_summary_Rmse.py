import pandas as pd
import matplotlib.pyplot as plt

rbf_summary = pd.read_csv('rbf_summary_results.csv')
poly_summary = pd.read_csv('polynomial_summary_results.csv')
linear_summary = pd.read_csv('linear_summary_results.csv')

summary_df = pd.DataFrame({
    'Metric': rbf_summary['Metric'],
    'RBF Min': rbf_summary['Min'],
    'RBF Max': rbf_summary['Max'],
    'Polynomial Min': poly_summary['Min'],
    'Polynomial Max': poly_summary['Max'],
    'Linear Min': linear_summary['Min'],
    'Linear Max': linear_summary['Max']
})

#Plot 1: Combined summary metrics 
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.12
metrics = summary_df['Metric']
indices = range(len(metrics))

# Plot mean bars
ax.bar([i - 2*bar_width for i in indices], summary_df['RBF Min'], width=bar_width, label='RBF Min')
ax.bar([i - bar_width for i in indices], summary_df['Polynomial Min'], width=bar_width, label='Polynomial Min')
ax.bar(indices, summary_df['Linear Min'], width=bar_width, label='Linear Min')

# Plot max bars
ax.bar([i + bar_width for i in indices], summary_df['RBF Max'], width=bar_width, label='RBF Max', alpha=0.5)
ax.bar([i + 2*bar_width for i in indices], summary_df['Polynomial Max'], width=bar_width, label='Polynomial Max', alpha=0.5)
ax.bar([i + 3*bar_width for i in indices], summary_df['Linear Max'], width=bar_width, label='Linear Max', alpha=0.5)

ax.set_xticks(indices)
ax.set_xticklabels(metrics)
ax.set_ylabel('Error')
ax.set_title('Combined Summary Metrics (Min and Max) Across Kernels')
ax.legend()
plt.tight_layout()
plt.show()

# Load detailed results
rbf_detailed = pd.read_csv('rbf_detailed_results.csv')
poly_detailed = pd.read_csv('polynomial_detailed_results.csv')
linear_detailed = pd.read_csv('linear_detailed_results.csv')

#Plot 2: Detailed RMSE for each load bus 
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rbf_detailed['Load Bus'], rbf_detailed['RMSE'], label='RBF RMSE', marker='o')
ax.plot(poly_detailed['Load Bus'], poly_detailed['RMSE'], label='Polynomial RMSE', marker='x')
ax.plot(linear_detailed['Load Bus'], linear_detailed['RMSE'], label='Linear RMSE', marker='s')
ax.set_xlabel('Load Bus')
ax.set_ylabel('RMSE')
ax.set_title('Detailed RMSE Across Load Buses for Different Kernels in Guassian Process')
ax.legend()
plt.tight_layout()
plt.show()
