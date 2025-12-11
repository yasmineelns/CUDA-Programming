# plot results from errors.csv in logarithmic scale
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('errors.csv', delimiter=',', skiprows=1)
N = data[:, 0]
F_est = data[:, 1]
AbsError = data[:, 2]
plt.figure(figsize=(10, 6))
plt.loglog(N, AbsError, marker='o', label='Absolute Error')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Absolute Error')
plt.title('View Factor Estimation Error vs Number of Samples')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('view_factor_error_plot.png')
plt.show()