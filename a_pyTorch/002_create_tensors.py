import torch

# 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4])

# 2D tensor
tensor_2d = torch.tensor([[1, 2], [3, 4]])

print(tensor_1d)
print(tensor_2d)

print("--------------------------------------------------------")

# Zeros tensor
zeros = torch.zeros(3, 2)  # 3x2 tensor of zeros

# Ones tensor
ones = torch.ones(2, 2)    # 2x2 tensor of ones

# Random tensor
random = torch.rand(3, 3)  # 3x3 tensor of random numbers between 0 and 1

print(zeros)
print(ones)
print(random)

print("--------------------------------------------------------")
# Integer tensor
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

# Float tensor
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

print(int_tensor)
print(float_tensor)

print("--------------------------------------------------------")

# Create a tensor from another tensor
like_tensor = torch.zeros_like(tensor_2d)

print(like_tensor)
print("--------------------------------------------------------")

# Create a sequence
sequence = torch.arange(0, 10, step=2)

# Create evenly spaced numbers over a specified interval
linspace = torch.linspace(0, 10, steps=5)

print(sequence)
print(linspace)

print("--------------------------------------------------------")
# Create a tensor
original = torch.tensor([1, 2, 3, 4, 5, 6])

# Reshape it
reshaped = original.reshape(2, 3)

print(original)
print(reshaped)
print("--------------------------------------------------------")



