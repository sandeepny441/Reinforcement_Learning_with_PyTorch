import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
input_size = 10
hidden_size = 20
output_size = 2
learning_rate = 0.01
num_epochs = 100

# Create dummy data
X = torch.randn(100, input_size)
y = torch.randint(0, output_size, (100,))

# Initialize the model
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.randn(1, input_size)
    prediction = model(test_input)
    _, predicted_class = torch.max(prediction, 1)
    print(f'Predicted class for test input: {predicted_class.item()}')