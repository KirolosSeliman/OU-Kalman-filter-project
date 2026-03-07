import numpy as np
import matplotlib.pyplot as plt


def random_walk(N, drift, sigma):
    shocks = np.random.normal(loc=drift, scale=sigma,size=N)
    price = np.zeros(N)
    price[0] = 100  # starting price
    for t in range(1, N):
        price[t] = price[t-1] + shocks[t]
    return price


for i in range(100):        # drift is like the slope of the random walk, like an extra push to make it lean in a certain way
    path = random_walk(N=200, drift=.1, sigma=2)
    plt.plot(path, alpha=0.3, linewidth=0.8)

plt.title("100 Random Walks")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()