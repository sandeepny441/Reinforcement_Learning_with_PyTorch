import torch

# Scalar (0D tensor)
scalar = torch.tensor(5)

# Vector (1D tensor)
vector = torch.tensor([1, 2, 3])

# Matrix (2D tensor)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 3D tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


print(scalar)

print(vector)

print(matrix)

print(tensor_3d)