#!/usr/bin/env python3
"""Program to generate a bifurcation diagram of the logistic map."""
import logistic_functions as lf
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Initial conditions and allocations
x_in = 0.01
# Specified r values
r_vals = np.linspace(2.9, 4, 10 ** 3)
# Number of iterations to arrive at the converged values
n_times = 10 ** 6
# Number of outputs to store
n_outs = 3000
# Array to store outputs after the map has converged to its final values
stable_outputs = np.zeros(n_outs, dtype=np.float64)
# Array to store the stabe outputs for each value of the parameter r
stored_outputs = np.zeros((len(stable_outputs), len(r_vals)), dtype=np.float64)


# Start the clock
tic = time.time()
# Sweep through the values of the parameter r, storing the converged outputs
lf.sweep_parameter_r(r_vals, x_in, n_times, stable_outputs,
                     stored_outputs)
# Stop the clock
toc = time.time()
# Print the time taken to generate the data
print("wall clock time = %f" % (toc - tic))


# Start the clock again
tic = time.time()
# Define a matplotlib figure to plot the bifurcation diagram
fig, ax = plt.subplots(1, figsize=(3.175, 2))
# Adjust the padding to make the subplots fill the space
fig.subplots_adjust(top=0.975, bottom=0.14, left=0.1, right=0.9)
# Define a matplotlib colormap
colors = plt.cm.YlGnBu(np.linspace(0.2, 1, len(r_vals)))
# Sweep through each value of the parameter r
for i in range(len(r_vals)):
    # Plot the converged outputs for each value of r
    ax.plot(np.ones(stored_outputs.shape[0], dtype=np.float64) * r_vals[i],
            stored_outputs[:, i], '.', markersize=0.1, color=colors[i],
            mew=0.1, alpha=0.1)


# Fiddle with the plot parameter to make it better
ax.set_xlim(left=min(r_vals), right=max(r_vals))
ax.set_ylim(bottom=0, top=1)
ax.minorticks_on()
ax.set_xlabel(r"$r$", fontsize=10, labelpad=0)
ax.set_ylabel(r"$x_{n}$", fontsize=10, labelpad=0)
ax.tick_params(which='both', right=True, top=True, direction='in',
               labelsize=8)
# Save the plot to file
fig.savefig('bifurcation.png', format='png', dpi=1000)
# Stop the clock
toc = time.time()
# Print the time taken to plot
print("plot time = %f" % (toc - tic))
