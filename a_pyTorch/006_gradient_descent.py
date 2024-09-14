import torch

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.weight * x + self.bias

# Create some dummy data
X = torch.linspace(-10, 10, 100)
y = 2 * X + 1 + torch.randn(100) * 0.5

# Instantiate the model and define loss function and optimizer
model = SimpleModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print final parameters
print(f"\nFinal parameters:")
print(f"Weight: {model.weight.item():.4f}")
print(f"Bias: {model.bias.item():.4f}")

# Visualize the results
import matplotlib.pyplot as plt

plt.scatter(X.numpy(), y.numpy(), label='Data')
plt.plot(X.numpy(), model(X).detach().numpy(), color='red', label='Fitted Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()