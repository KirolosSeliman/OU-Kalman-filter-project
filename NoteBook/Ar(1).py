import numpy as np
import matplotlib.pyplot as plt



# X_t = X_{t-1} + (1 - φ) · (μ - X_{t-1})
# where: (μ - X_{t-1}), is the price distance from the mean
# (1 - φ): is the push that we add to get to the mean
#ex: X_{t-1} = 200, μ= 203, φ = 0.84
# next price is (with factoring randomness) = (1-0.84) * (203-200)= 0.48 incrment up

def Ar1(phi, mu, sigma, N, X0):
    shocks = np.random.normal(scale=sigma, size=N)
    price = np.zeros(N)
    price[0] = X0
    for t in range(1, N):                              # shocks is a number from a normal distribution that is added or substracted
        price[t] = phi * price[t-1] + (1 - phi) * mu + shocks[t]
    return price 


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("AR(1) Process — How φ Controls Mean Reversion", fontsize=14)

configs = [
    {"phi": 0.5,  "title": "φ = 0.50 — Fast mean reversion"},
    {"phi": 0.84, "title": "φ = 0.84 — Your strategy's value"},
    {"phi": 0.99, "title": "φ = 0.99 — Almost a random walk"},
    {"phi": 1.0,  "title": "φ = 1.00 — Pure random walk"},
]

for ax, config in zip(axes.flatten(), configs):
    for i in range(30):
        path = Ar1(phi=config["phi"], mu=100, sigma=1, N=200, X0=80)
        ax.plot(path, alpha=0.3, linewidth=0.8, color="steelblue")
    
    ax.axhline(y=100, color="red", linewidth=1.5, linestyle="--", label="Mean (μ=100)")
    ax.set_title(config["title"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

plt.tight_layout()
plt.show()
