import numpy as np
from tqdm import tqdm

# Container data
multipliers = np.array([10, 17, 20, 31, 37, 50, 73, 80, 89, 90], dtype=float)
inhabitants = np.array([1, 1, 2, 2, 3, 4, 4, 6, 8, 10], dtype=float)

# Initialize number of picks per container
picks = np.zeros(10)

for num_choices in tqdm(range(1, 100000)):  # simulate until convergence
    # Compute value of each container based on current load
    value = multipliers / (inhabitants + picks / num_choices * 100)

    # Assign choices greedily
    top = np.argsort(value)[-1]
    picks[top] += 1

print([float(f'%.5f' % x) for x in picks/np.sum(picks)])

value = multipliers / (inhabitants + picks / np.sum(picks) * 100)

print([float(f'%.5f' % x) for x in value])