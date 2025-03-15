import matplotlib.pyplot as plt
import numpy as np
import yaml as yml
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def advance_schedule(timesteps, scale_start, scale_end, width):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    return alphas_cumprod

def segment_schedule(timesteps, time_segment, segment_diff, **kwargs):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        alphas_this = advance_schedule(time_this, **params)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)
    return alphas_cumprod


with open("1.yaml", "r") as f:
    config = yml.safe_load(f)

t = np.linspace(0, 1000, 1000) 
node_alhpas = advance_schedule(1000,0.9999,0.0001,3)
bond_alhpas = segment_schedule(1000,config["time_segment"],config["segment_diff"])

plt.plot(t, node_alhpas, label="atom/pos", color="blue")
plt.plot(t, bond_alhpas, label="bond", color="red")
plt.xlabel("Timesteps")
plt.ylabel("Noise Level")
plt.legend()
plt.grid(True)
plt.savefig("1.png", dpi=300)