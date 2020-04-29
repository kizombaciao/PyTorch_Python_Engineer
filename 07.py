# 1. Design model (input, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)       # pytorch method to reshape tensors

n_samples, n_features = X.shape

# 1) model



# 2) loss / optimizer

# 3) training loop


