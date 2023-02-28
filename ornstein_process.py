#############################################################
#       SIMULTATING STOCHASTIC DIFFERENTIAL EQUATION        #
#############################################################

# Title of problem:
# Simulate Ornstein-Unlenbeck process, which is a solution of Langevin equation.
# Model:
# dx = - (x - mu) / tau * dt + sigma * sqrt(2/tau) * dW
# Elements of model:
# x(t): stochastic process
# dx: infinitesimal increment
# mu: mean
# sigma: standard deviation
# tau: time constant
# W: brownian motion term (from Wienner process)

# import packages
import numpy as np
import matplotlib.pyplot as plt

# define function 

def main():
    mu = 10         # mean
    sigma = 1       # standard deviation
    tau = 0.05      # time constant
    T = 1           # total time
    dt = 0.001      # time step
    n = int(T/dt)   # T = n * dt 

    # build vector of times
    t = np.linspace(start = 0.0, stop = T, num = n)

    # vector of sucessive values of process
    x = np.zeros(n)

    # compute the unchanged parts of equation 
    # x_(n+1) = x_(n) + f(x, t) dt + g(x, t) * sqrt(dt) * eta    
    g_xt = sigma * np.sqrt(2.0 / tau)
    sqrt_dt = np.sqrt(dt)

    # simulation using Euler-Maruyama method
    # for random variable with mean = 0 and std = 1 ===>  eta = 0 + np.random.randn()
    for i in range(n-1):
        f_xt = -(x[i] - mu)/tau
        x[i+1] = x[i] + f_xt * dt + g_xt * sqrt_dt * np.random.randn()
    
    # plot the evolution of the process
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    ax.plot(t, x, lw = 3)
    plt.show()

if __name__ == "__main__":
    main()


