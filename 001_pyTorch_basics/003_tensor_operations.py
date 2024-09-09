print("-------Arithmetic-----------")
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

print("-------Multiplication-----------")
m1 = torch.tensor([[1, 2], [3, 4]])
m2 = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
print(torch.mm(m1, m2))
# or
print(m1 @ m2)

print("------------slicing--------------")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Get the first row
print(x[0])

# Get a specific element
print(x[1, 2])  # Element at 2nd row, 3rd column

# Slicing
print(x[:2, 1:])  # First two rows, second column onwards

