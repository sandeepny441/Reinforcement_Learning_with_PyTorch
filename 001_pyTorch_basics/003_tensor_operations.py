print("-==========-Arithmetic-===========-")
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(a + b)  # or torch.add(a, b)

# Subtraction
print(a - b)  # or torch.sub(a, b)

# Multiplication (element-wise)
print(a * b)  # or torch.mul(a, b)

# Division (element-wise)
print(a / b)  # or torch.div(a, b)

# Operations with scalars
print(a + 2)
print(a * 3)

print("=========--Multiplication--==========-")
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
print(torch.mm(m1, m2))
# or
print(m1 @ m2)

print("-=========--Indexing and Slicing-========--")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Get the first row
print(x[0])

# Get a specific element
print(x[1, 2])  # Element at 2nd row, 3rd column

# Slicing
print(x[:2, 1:])  # First two rows, second column onwards

print("--============-Reshaping and Transposing=========")
y = torch.tensor([1, 2, 3, 4, 5, 6])

# Reshape
print(y.view(2, 3))
# or
print(y.reshape(3, 2))

# Transpose
z = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(z.t())

print("==========concat============")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along dimension 0 (vertically)
print(torch.cat((a, b), dim=0))

# Concatenate along dimension 1 (horizontally)
print(torch.cat((a, b), dim=1))

print("============-element wise functions===========-")
x = torch.tensor([-1, 0, 1, 2])

print(torch.abs(x))    # Absolute value
print(torch.exp(x))    # Exponential
print(torch.log(torch.abs(x) + 1))  # Logarithm

print("============Reduction Operations===========")
y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(y.sum())         # Sum of all elements
print(y.mean())        # Mean of all elements
print(y.max())         # Maximum value
print(y.argmax())      # Index of maximum value
