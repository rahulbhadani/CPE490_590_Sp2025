# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x = np.random.uniform(30, 100, 50)  # Random x values
y = 4.0 + 3.0 * x + np.random.normal(0, 10, 50)  # y = 4 + 3x + Gaussian noise

# Convert data to PyTorch tensors
X = torch.tensor(x, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature and one output

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegression()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Train the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get the learned parameters (weights and bias)
weight = model.linear.weight.item()
bias = model.linear.bias.item()

# Print the learned parameters
print(f'Learned weight: {weight:.4f}')
print(f'Learned bias: {bias:.4f}')

fig, ax = plt.subplots(figsize=(17, 14))
# Plot the results
plt.scatter(x, y, label='Original data')
plt.plot(x, weight * x + bias, label='Fitted line', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

