from sklearn.linear_model import LogisticRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # User engagement score
y = np.array([0, 0, 0, 1, 1])            # Conversion (0 = No, 1 = Yes)

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict conversion for a new user engagement score
new_engagement = np.array([[3.5]])
predicted_conversion = model.predict(new_engagement)
print("Predicted Conversion:", predicted_conversion)
