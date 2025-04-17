import numpy as np
from tqdm import tqdm

# Container data
multipliers = np.array([10, 17, 20, 23, 30, 31, 37, 40, 41, 47, 50, 60, 70, 73, 79, 80, 83, 89, 90, 100], dtype=float)
inhabitants = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 10, 15], dtype=float)

# Initialize number of picks per container
picks = np.zeros(20) + 0.0000000001

for num_choices in tqdm(range(1, 100000)):  # simulate until convergence
    # Compute value of each container based on current load
    value = multipliers / (inhabitants + picks / num_choices * 100)

    # Assign choices greedily
    top = np.argsort(value)[-1]
    #picks[top] += 1

print([float(f'%.3f' % x) for x in multipliers])
print([float(f'%.3f' % x) for x in picks/np.sum(picks)])

value = multipliers / (inhabitants + picks / np.sum(picks) * 100)

print([float(f'%.3f' % x) for x in value])