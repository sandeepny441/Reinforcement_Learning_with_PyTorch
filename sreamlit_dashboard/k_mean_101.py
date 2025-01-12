import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.max_iters):
            model = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(model)
        return np.round(y_pred)

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)  # 300 points with 2 features

# K-Means clustering function
def kmeans(X, n_clusters, max_iters=100, tol=1e-4):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids

# Run K-Means clustering
n_clusters = 4
labels, centroids = kmeans(X, n_clusters)

# Convert cluster labels to binary targets for simplicity
binary_labels = (labels == 0).astype(int)

# Train Logistic Regression
log_reg = LogisticRegression(learning_rate=0.1, max_iters=1000)
log_reg.fit(X, binary_labels)

# Predict using Logistic Regression
predictions = log_reg.predict(X)

# Plot original clusters and logistic regression results
plt.figure(figsize=(12, 6))

# Original clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r', label='Centroids')
plt.title('Original K-means Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.legend()

# Logistic Regression predictions
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', alpha=0.7, edgecolor='k')
plt.title('Logistic Regression Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

plt.tight_layout()
plt.show()
