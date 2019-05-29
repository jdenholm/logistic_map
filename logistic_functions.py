"""Functions for iterating the logistic map."""
import numpy as np
from numba import jit, int64, float64


@jit(float64(int64, float64, float64), nopython=True)
def iterate_map(n_times, r, x_in):
    """Iterate the logistic map n_times."""
    # Variable to store the input value
    x0 = np.float64(x_in)
    # Variable to store the output value
    x1 = np.float64(0)
    # Iterate the map n_times
    for i in range(n_times):
        # Get the output value based on current input value
        x1 = r * x0 * (1 - x0)
        # Feed the output value back as the input value
        x0 = x1
    # Return the output value
    return(x1)


@jit((float64[:], float64, float64), nopython=True)
def store_stable_solutions(stable_outputs, r, x_in):
    """Save the len(stable_outputs) number of outputs."""
    x0 = np.float64(x_in)
    # For each of the stable outputs to be stored
    for i in range(len(stable_outputs)):
        # Store the output value
        stable_outputs[i] = r * x0 * (1 - x0)
        # Feed the output back in as the input
        x0 = stable_outputs[i]
    return()


@jit((float64[:], float64, float64, float64[:], float64[:, :]), nopython=True)
def sweep_parameter_r(r_vals, x_in, n_times, stable_outputs,
                      stored_outputs):
    """Generate the bifurcation diagram of the logistic map."""
    # Store the input value
    x0 = np.float64(x_in)
    # Sweep through each value of the parameter r
    for i in range(len(r_vals)):
        # Set the input value
        x0 = np.float64(x_in)
        # Iterate the map n_times
        x_out = iterate_map(n_times, r_vals[i], x0)
        # Feed output back in as input
        x_in = x_out
        # Store the stable solutions
        store_stable_solutions(stable_outputs, r_vals[i], x0)
        # Store the outputs for this value of the parameter r
        stored_outputs[:, i] = stable_outputs.copy()
    return()
