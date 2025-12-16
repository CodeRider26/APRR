import os
import numpy as np
from PIL import Image
import os

np.random.seed(42)

data_dir = 'data'
result_data_dir = './draw_data'

if not os.path.exists(result_data_dir):
    os.makedirs(result_data_dir)

image_files = [os.path.join(data_dir, f'{i}.jpg') for i in range(1, 6)]
images = []
for fpath in image_files:
    img = Image.open(fpath).convert('L')  
    img = img.resize((28, 28)) 
    img_arr = np.array(img, dtype=np.float64).reshape(-1) / 255.0  
    images.append(img_arr)

true_x = images[0]
n_features = true_x.shape[0]  # 784
m = 10  

measurement_dim = 100  
A = [np.random.randn(measurement_dim, n_features) for _ in range(m)]

y = [A[i] @ true_x for i in range(m)]

r = np.random.normal(0, 1e-3, measurement_dim)

initial_x = true_x + np.random.randn(n_features) * 2.0e-3

class PGRRAlgorithm:
    def soft_thresholding(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def PG_RR(self, initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C):
        x = initial_x.copy()
        n = len(y)
        y0 = y.copy()
        current_errs = []
        current_errs.append((np.linalg.norm((x - true_x))**2) / len(y))

        for iteration in range(num_epochs):
            y0 = [y0i + C * r for y0i in y0]  # Add error vector

            for i in np.random.permutation(n):
                gradient = 2 * A[i].T @ (A[i] @ x - y0[i])
                x = self.soft_thresholding(x - gamma * gradient, gamma * lambda_)

            if (iteration != 0) and (iteration % p == 0):
                err = (np.linalg.norm((x - true_x))**2) / len(y0)
                current_errs.append(err)
                print(f"[PGRR] Iteration {iteration}, Objective Value: {err:.6e}")

            y0 = y.copy()

        return x, current_errs

class OursAlgorithms:
    def soft_thresholding(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def PG_RR(self, initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C, momentum):
        x = initial_x.copy()
        n = len(y)
        velocity = np.zeros_like(x)
        y0 = y.copy()
        current_errs = []
        current_errs.append((np.linalg.norm((x - true_x))**2) / len(y))

        for iteration in range(num_epochs):
            y0 = [y0i + C * r for y0i in y0]

            for i in np.random.permutation(n):
                gradient = 2 * A[i].T @ (A[i] @ x - y0[i])
                velocity = momentum * velocity + gamma * gradient
                x = self.soft_thresholding(x - velocity, gamma * lambda_)

            if (iteration != 0) and (iteration % p == 0):
                err = (np.linalg.norm((x - true_x))**2) / len(y0)
                current_errs.append(err)
                print(f"[Ours m={momentum:.2f}] Iteration {iteration}, Objective Value: {err:.6e}")

            y0 = y.copy()

        return x, current_errs

lambda_ = 5e-5  
num_epochs = 60000  # total epochs
p = 3000  # record error every p epochs
gamma = 1.0e-7  # step size
C = 3  # error constant

momentum_values = [0.0, 0.5, 0.7, 0.9, 0.99]

# Prepare to save results
results = []

pgrr = PGRRAlgorithm()
x_pgrr, errs_pgrr = pgrr.PG_RR(initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C)
results.append({'filename': 'draw_data_000.npy', 'description': 'Baseline PGRR method without momentum', 'data': np.array(errs_pgrr)})

ours = OursAlgorithms()
for idx, mom in enumerate(momentum_values):
    if mom == 0.0:
        continue
    x_ours, errs_ours = ours.PG_RR(initial_x, A, y, lambda_, true_x, r, num_epochs, p, gamma, C, mom)
    filename = f'draw_data_{idx+1:03d}.npy'
    results.append({'filename': filename, 'description': f'Ours method with momentum={mom}', 'data': np.array(errs_ours)})

# Save data files
for res in results:
    filepath = os.path.join(result_data_dir, res['filename'])
    np.save(filepath, res['data'])
