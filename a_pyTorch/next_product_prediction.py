import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast

# Convert 'id_date' to datetime and sort by 'atg_id' and 'id_date'
df['id_date'] = pd.to_datetime(df['id_date'])
df = df.sort_values(by=['atg_id', 'id_date'])

# Convert string lists into actual lists, if necessary
df['products'] = df['products'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Initialize LabelEncoder to convert product IDs to integers
product_encoder = LabelEncoder()
all_products = [item for sublist in df['products'] for item in sublist]
product_encoder.fit(all_products)
df['products'] = df['products'].apply(lambda x: product_encoder.transform(x))

# Define fixed length for individual product lists
inner_pad_length = 13  # Adjust as needed

# Pad individual product lists
def pad_inner_list(product_list, pad_length):
    product_list = list(product_list)
    if len(product_list) < pad_length:
        return product_list + [0] * (pad_length - len(product_list))  # Assuming 0 as the pad token
    else:
        return product_list[:pad_length]

df['products'] = df['products'].apply(lambda x: pad_inner_list(x, inner_pad_length))

# Generate sequences and targets for model input
sequence_length = 5
sequences = []
targets = []
for atg_id, group in df.groupby('atg_id'):
    product_lists = group['products'].tolist()
    for i in range(len(product_lists) - sequence_length):
        seq = product_lists[i:i+sequence_length]
        target = product_lists[i+sequence_length]
        sequences.append(torch.tensor(seq, dtype=torch.long))
        targets.append(torch.tensor(target, dtype=torch.long))

# Pad sequences and targets
sequences_padded = pad_sequence(sequences, batch_first=True)
targets_padded = pad_sequence(targets, batch_first=True)

# Split data into training and testing sets
train_seqs, test_seqs, train_tgts, test_tgts = train_test_split(sequences_padded, targets_padded, test_size=0.2, random_state=42)

# Define custom Dataset and DataLoader
class ProductDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

train_dataset = ProductDataset(train_seqs, train_tgts)
test_dataset = ProductDataset(test_seqs, test_tgts)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class ProductPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ProductPredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output for prediction
        return out

# Initialize model, loss function, and optimizer
vocab_size = len(product_encoder.classes_)
embed_size = 128
hidden_size = 256
num_layers = 2
model = ProductPredictionModel(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
# Train the model without reshaping the sequences, as they should already be 3D
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for sequences, targets in train_loader:
        # Ensure sequences are in (batch_size, sequence_length, embed_size)
        
        # Forward pass
        outputs = model(sequences)  # Model expects (batch_size, sequence_length, input_size)
        loss = criterion(outputs, targets)  # Adjust criterion as needed
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequences, targets in test_loader:
        sequences = sequences.view(sequences.size(0), sequences.size(1), -1)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
