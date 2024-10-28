import torch
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    # Randomly initialize cluster centers
    n_samples, n_features = X.shape
    centroids = X[torch.randperm(n_samples)[:k]]
    
    for _ in range(max_iters):
        # Assign samples to nearest centroid
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(k)])
        
        # Check for convergence
        if torch.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Generate sample data
n_samples = 300
X = torch.cat([
    torch.randn(n_samples // 3, 2) + torch.tensor([2, 2]),
    torch.randn(n_samples // 3, 2) + torch.tensor([-2, -2]),
    torch.randn(n_samples // 3, 2) + torch.tensor([2, -2])
])

# Perform K-means clustering
k = 3
labels, centroids = kmeans(X, k)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering Results')
plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt


import torch
import numpy as np
import matplotlib.pyplot as plt


import torch
import numpy as np
import matplotlib.pyplot as plt