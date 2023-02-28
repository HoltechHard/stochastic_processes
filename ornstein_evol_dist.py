####################################################################
#       EVOLUTION DISTRIBUTION OF PROB. FOR STOCHASTIC PROCESS     #
####################################################################

# import packages
import numpy as np
import matplotlib.pyplot as plt

def main():
    mu = 10         # mean
    sigma = 1       # standard deviation
    tau = 0.05      # time constant
    T = 1           # total time
    dt = 0.001      # time step
    n = int(T/dt)   # T = n * dt 

    ntrials = 10000
    X = np.zeros(ntrials)

    bins = np.linspace(start = -2.0, stop = 14.0, num = 100)
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))

    # compute the unchanged parts of equation 
    # x_(n+1) = x_(n) + f(x, t) dt + g(x, t) * sqrt(dt) * eta    
    g_xt = sigma * np.sqrt(2.0 / tau)
    sqrt_dt = np.sqrt(dt)

    for i in range(n):
        f_xt = -(X - mu)/tau
        X += f_xt * dt + g_xt * sqrt_dt * np.random.randn(ntrials)

        # show the prob. dist. for some points
        if i in (5, 50, 100, 500, 900):
            hist, _ = np.histogram(X, bins = bins)
            ax.plot((bins[1:] + bins[:-1])/2, hist, 
                    {5: '-', 50: '-', 100: '-', 500: '-', 900: '-', }[i],
                    label = f"t={i * dt:.3f}")
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()


