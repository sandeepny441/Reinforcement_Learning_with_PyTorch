import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Create a custom Dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.randn(size, 5)  # 1000 samples, 5 features each
        self.labels = np.random.randint(0, 2, size)  # Binary labels (0 or 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# Step 2: Create an instance of the dataset
dataset = SimpleDataset()

# Step 3: Create a DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 4: Iterate through the data
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Print first few items in the batch
    print(f"  First few data points: \n{data[:2]}")
    print(f"  First few labels: {labels[:5].squeeze()}")
    
    # Break after 3 batches for this example
    if batch_idx == 2:
        break

print("\nDataLoader iteration complete!")