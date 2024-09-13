import torch

# Create tensors with requires_grad=True to track computations
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0], requires_grad=True)

# Perform some operations
z = x**2 + y**3

# Compute the sum of all elements in z
loss = z.sum()

# Backward pass: compute gradients
loss.backward()

# Print the gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

# Create a more complex computation graph
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 2.0], requires_grad=True)
c = a**2 + b**3
d = c.mean()

# Compute gradients
d.backward()

print("\nGradient of a:", a.grad)
print("Gradient of b:", b.grad)

# Disable gradient tracking
with torch.no_grad():
    x = x * 2
    y = y + 1
    print("\nOperations without gradient tracking:")
    print("x:", x)
    print("y:", y)

# Reset gradients
x.grad.zero_()
y.grad.zero_()

print("\nReset gradients:")
print("x.grad:", x.grad)
print("y.grad:", y.grad)