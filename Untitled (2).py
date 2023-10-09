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
#A2 Bipolar Function
# Define Bi-Polar Step function
def bipolar_step_function(x):
    return np.where(x >= 0, 1, -1)#Bipolar function has values -1,0,1
converge_error = 0.002#convergence error value is given in the question
alpha = 0.05
epochs = 1000
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])#inputs for and gate
y = np.array([0, 0, 0, 1])#outputs for and gate
W = np.array([10, 0.2, -0.75])#weights given in questio 
errorlist = []#initialising an error list
for epoch in range(epochs):
    error = 0#initialise error value to 0
    for i in range(len(X)):
        y_pred = bipolar_step_function(np.dot(X[i], W[1:]) + W[0])
        W[1:] += alpha * (y[i] - y_pred) * X[i]
        W[0] += alpha * (y[i] - y_pred)
        error += (y[i] - y_pred) ** 2
    error /= len(X)#calculating error after each iteration
    errorlist.append(error)#Append the error list with the error gained after each iteration
    if error <= converge_error:
        print("Bi-Polar Step Converged after", epoch+1, "epoch")#Printing after how many iterations ReLU function converged
        break
plt.plot(range(1, epoch+2), errorlist)
print("Bi-Polar Step Converged after", epoch+1, "epoch")
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs Epoch')
plt.show()#Plotting the graph for ReLU function

#A2 Sigmoid Function
#defining a sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
converge_error = 0.002#convergence error to be taken is given in the question
errorlist = []#initialise an error list
alpha = 0.05
epochs = 1000
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])#inputs for and gate
y = np.array([0, 0, 0, 1])#outputs for and gate 
W = np.array([10, 0.2, -0.75])#weights to be taken given in the question
for epoch in range(epochs):
    error = 0#initialise error value to 0
    for i in range(len(X)):
        y_pred = sigmoid(np.dot(X[i], W[1:]) + W[0])  
        W[1:] += alpha * (y[i] - y_pred) * X[i]
        W[0] += alpha * (y[i] - y_pred)  
        error += (y[i] - y_pred) ** 2
    error /= len(X)#calculate error for each iteration
    errorlist.append(error)#Append the error list with the error gained after each iteration
    if error <= converge_error:
        print("Sigmoid Function Converged after", epoch + 1, "epoch")#Printing after how many iterations ReLU function converged
        break
plt.plot(range(1, epoch + 2), errorlist)
print(f"Sigmoid Converged after {epoch+1} epoch")
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs Epoch (Sigmoid)')
plt.show()#Plotting the graph for ReLU function
#A2 Relu function
# Define ReLU function
def relu(x):
    return np.maximum(0, x)
converge_error = 0.002#error value to be taken is given in the question
alpha = 0.05
epochs = 1000
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])#input values for and gate
y = np.array([0, 0, 0, 1])#output values for and gate
W = np.array([10, 0.2, -0.75])#weights given in the question
errorlist = []
for epoch in range(epochs):
    error = 0#initialise error to 0
    for i in range(len(X)):
        y_pred = relu(np.dot(X[i], W[1:]) + W[0])
        W[1:] += alpha * (y[i] - y_pred) * X[i]
        W[0] += alpha * (y[i] - y_pred)
        error += (y[i] - y_pred) ** 2
    error /= len(X)#Calculating error in each iteration
    errorlist.append(error)#Appending the array with the error gained
    if error <= converge_error:
        print("ReLU Converged after", epoch+1, "epoch")#Printing after how many iterations ReLU function converged
        break
plt.plot(range(1, epoch+2), errorlist)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs Epoch')
plt.show()#Plotting the graph for ReLU function









