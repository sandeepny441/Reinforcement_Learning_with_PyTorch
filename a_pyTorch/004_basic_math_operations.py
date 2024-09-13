import torch

# Create some tensors to work with
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])

# 1. Addition
print("Addition:")
print(a + b)  # Element-wise addition
print(torch.add(a, b))  # Same as above, using torch.add()

# 2. Subtraction
print("\nSubtraction:")
print(b - a)  # Element-wise subtraction
print(torch.sub(b, a))  # Same as above, using torch.sub()

# 3. Multiplication
print("\nMultiplication:")
print(a * b)  # Element-wise multiplication
print(torch.mul(a, b))  # Same as above, using torch.mul()

# 4. Division
print("\nDivision:")
print(b / a)  # Element-wise division
print(torch.div(b, a))  # Same as above, using torch.div()

# 5. Matrix multiplication
print("\nMatrix multiplication:")
print(torch.matmul(c, d))  # Matrix multiplication
print(c @ d)  # Same as above, using @ operator

# 6. Power
print("\nPower:")
print(a ** 2)  # Element-wise power
print(torch.pow(a, 2))  # Same as above, using torch.pow()

# 7. Square root
print("\nSquare root:")
print(torch.sqrt(a))

# 8. Exponential and logarithm
print("\nExponential and logarithm:")
print(torch.exp(a))
print(torch.log(a))

# 9. Sum and mean
print("\nSum and mean:")
print(torch.sum(a))
print(torch.mean(a))

# 10. Min and max
print("\nMin and max:")
print(torch.min(a))
print(torch.max(a))