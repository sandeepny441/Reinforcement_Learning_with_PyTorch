import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate some sample data
x = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + torch.randn(x.shape) * 0.5

# Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Plot the results
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), model(x).detach().numpy(), color='r', label='Fitted Line')
plt.legend()
plt.show()

# Print the learned parameters
print(f"Learned parameters: w = {model.linear.weight.item():.2f}, b = {model.linear.bias.item():.2f}")

print("this is a new update")


# 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate some sample data
x = torch.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + torch.randn(x.shape) * 0.5

# Define the model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Plot the results
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), model(x).detach().numpy(), color='r', label='Fitted Line')
plt.legend()
plt.show()

# Print the learned parameters
print(f"Learned parameters: w = {model.linear.weight.item():.2f}, b = {model.linear.bias.item():.2f}")

print("this is a new update")