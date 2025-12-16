import os

import numpy as np
from PIL import Image
import os

np.random.seed(42)

data_path = './draw_data'

os.makedirs(data_path, exist_ok=True)

image_files = ['./data/1.jpg', './data/2.jpg', './data/3.jpg', './data/4.jpg', './data/5.jpg']
images = []
for f in image_files:
    img = Image.open(f).convert('L') 
    img = img.resize((28, 28))  
    img_arr = np.array(img).reshape(-1) / 255.0  
    images.append(img_arr)

true_x = images[3]

noise_std = 3e-3
initial_x = true_x + noise_std * np.random.randn(len(true_x))

m = 10  
n_features = len(true_x)  
phi_rows = 100

O = np.random.randn(m, phi_rows, n_features)
A = [O[i] for i in range(m)]
y = [A[i] @ true_x for i in range(m)]


r = np.random.normal(loc=0, scale=1e-3, size=phi_rows)

lambda_ = 1e-5
C_values = [3]
gamma = 6.5e-8  # step size
momentum = 0.9
num_epochs = 50000  # number of iterations for epoch-based runs
p = 5000  # interval for recording error


class PGRRAlgorithm:
    def soft_thresholding(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def PG_RR(self, initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C):
        x = initial_x.copy()
        n = len(y)
        current_errs = []
        current_errs.append((np.linalg.norm((x - true_x)) ** 2) / len(y))

        for iteration in range(num_epochs):
            y0 = [y[i] + C * r for i in range(n)]

            for i in np.random.permutation(n):
                gradient = 2 * A[i].T @ (A[i] @ x - y0[i])
                x = self.soft_thresholding(x - gamma * gradient, gamma * lambda_)

            if (iteration != 0) and (iteration % p == 0):
                err = (np.linalg.norm((x - true_x)) ** 2) / len(y)
                current_errs.append(err)
                print(f"Baseline Iteration {iteration}, Objective Value: {err}")

        return x, current_errs

class OursAlgorithm:
    def soft_thresholding(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def PG_RR(self, initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C, momentum):
        x = initial_x.copy()
        n = len(y)
        velocity = np.zeros_like(x)
        current_errs = []
        current_errs.append((np.linalg.norm((x - true_x)) ** 2) / len(y))

        for iteration in range(num_epochs):
            y0 = [y[i] + C * r for i in range(n)]

            for i in np.random.permutation(n):
                gradient = 2 * A[i].T @ (A[i] @ x - y0[i])
                velocity = momentum * velocity + gamma * gradient
                x = self.soft_thresholding(x - velocity, gamma * lambda_)

            if (iteration != 0) and (iteration % p == 0):
                err = (np.linalg.norm((x - true_x)) ** 2) / len(y)
                current_errs.append(err)
                print(f"Ours Iteration {iteration}, Objective Value: {err}")

        return x, current_errs

results = []
for idx, C in enumerate(C_values):
    print(f"\nRunning experiments for noise constant C = {C}")

    # Baseline
    base_model = PGRRAlgorithm()
    baseline_x, baseline_errors = base_model.PG_RR(
        initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C
    )

    # Ours
    our_model = OursAlgorithm()
    our_x, our_errors = our_model.PG_RR(
        initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C, momentum
    )

    # Save errors to file
    baseline_filename = os.path.join(data_path, f'draw_data_base.npy')
    our_filename = os.path.join(data_path, f'draw_data_our.npy')

    np.save(baseline_filename, np.array(baseline_errors))
    np.save(our_filename, np.array(our_errors))
