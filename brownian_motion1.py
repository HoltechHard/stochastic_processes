##############################################
#    BROWNIAN-MOTION SIMULATION IN PYTHON    #
##############################################

import numpy as np
import matplotlib.pyplot as plt

# quadratic variation of stochastic process
# [Xt] = sum_{k=1..n} {(Xt_k - Xt_k-1)^2}
# theoretically: 
# [Xt] = t (linear)

def quadratic_variation(B):
    return np.cumsum(np.power(np.diff(B, axis = 0, prepend = 0.0), 2), axis = 0)

def main():
    # number of iterations
    n = 100
    # period
    T = 1.0
    # add multiple samples of brownian-motion
    d = 1000

    # parametral space
    times = np.linspace(0, T, n)
    
    # time step
    dt = times[1] - times[0]
    
    # property of independently increments
    # prop.1 => B0 = 0
    B0 = np.zeros(shape = (1, d))
    # prop.4 => Bt - Bs ~ N(0, t-s)
    # dB ~ N(0, dt)  ==> dB ~ sqrt(dt) * N(0, 1)
    dB = np.sqrt(dt) * np.random.normal(size = (n-1, d))
    # vector Bt ~ N(0, t)
    B = np.concatenate((B0, np.cumsum(dB, axis = 0)), axis = 0)
    plt.plot(times, B)
    plt.plot(times, quadratic_variation(B))
    plt.show()

if __name__ == "__main__":
    main()

