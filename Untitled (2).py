#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatal.xlsx")
df


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

 

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

 

# Separate features (X) and target (y)
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_3', 'embed_4']].values  # Convert DataFrame columns to a NumPy array
y = binary_df['Label'].values

 

# Split the data into training and test sets (70% training, 30% test)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

 

# Define the step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

 

# Initialize weights
weights = np.array([W0, W1, W2])

 

# Initialize the number of epochs and convergence condition
max_epochs = 1000
convergence_error = 0.002

 

# Lists to store epoch and error values for plotting
epoch_list = []
error_list = []

 

# Training loop
for epoch in range(max_epochs):
    error_sum = 0
    for i in range(len(Tr_X)):
        # Compute the weighted sum
        weighted_sum = np.dot(weights, np.insert(Tr_X[i], 0, 1))
        # Apply the step activation function
        prediction = step_activation(weighted_sum)
        # Compute the error
        error = Tr_y[i] - prediction
        error_sum += error
        # Update the weights
        weights += learning_rate * error * np.insert(Tr_X[i], 0, 1)

    # Calculate the sum-squared error
    mse = (error_sum ** 2) / len(Tr_X)

    # Append epoch and error values to lists for plotting
    epoch_list.append(epoch + 1)
    error_list.append(mse)

    # Print the error for this epoch
    print(f"Epoch {epoch + 1}: Mean Squared Error = {mse}")

    # Check for convergence
    if mse <= convergence_error:
        print("Convergence reached. Training stopped.")
        break

 

# Print the final weights
print("Final Weights:")
print(f"W0 = {weights[0]}, W1 = {weights[1]}, W2 = {weights[2]}")

 

# Plot epochs against error values
plt.figure(figsize=(8, 6))
plt.plot(epoch_list, error_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Epochs vs. Error")
plt.grid(True)
plt.show()



# In[ ]:




