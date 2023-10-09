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
#a3
import numpy as np
import matplotlib.pyplot as plt
# Initial weights
weights = np.array([10, 0.2, -0.75])
# XOR gate training data
training_data_xor = [
    {'input': np.array([1, 0, 0]), 'output': 0},
    {'input': np.array([1, 0, 1]), 'output': 1},
    {'input': np.array([1, 1, 0]), 'output': 1},
    {'input': np.array([1, 1, 1]), 'output': 0},
]
#learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
def step_function(x):
    return 1 if x >= 0 else 0
# Function to train the perceptron and return the convergence epoch
def train_perceptron(learning_rate):
    weight_copy = np.array(weights)  # Make a copy of initial weights
    convergence_epoch = None
    for epoch in range(2000):  # Maximum of 2000 epochs
        total_error = 0
        for data in training_data_xor:
            input_data = data['input']
            target_output = data['output']     
            # Calculate the weighted sum
            weighted_sum = np.dot(input_data, weight_copy)          
            # Apply the step activation function
            pred_output = step_function(weighted_sum)
            # Calculate the error
            error = target_output - pred_output       
            # Update the weights
            delta_w = learning_rate * error * input_data
            weight_copy += delta_w
            total_error += error
        # Check for convergence
        if abs(total_error) <= 0.002:
            convergence_epoch = epoch
            break
    return convergence_epoch
# Train the perceptron with different learning rates and record convergence epochs
convergence_epochs = []
for lr in learning_rates:
    convergence_epoch = train_perceptron(lr)
    convergence_epochs.append(convergence_epoch)
#Printing convergence epochs
print("CONVERGENCE EPOCHS FOR DIFFERENT LEARNING RATES:")
print(convergence_epochs)
# plot showing learning rates vs. convergence epochs
plt.plot(learning_rates, convergence_epochs, marker='o', linestyle='-', color='b')
plt.xlabel('Learning Rate')
plt.ylabel('Convergence Epoch')
plt.title('Learning Rate vs. Convergence Epoch')
plt.grid(True)
plt.show()

 

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

#A3
#define step function
def step(x):
  if x >= 0:
    return 1
  else:
    return 0
# Initialize the weights as given in the question
W0 = 10
W1 = 0.2
W2 = -0.75
#Initialise learning rate
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# Train the perceptron for different learning rates
iterations = []
for learning_rate in learning_rates:
  # Train the perceptron
  iteration_count = 0
  for i in range(1000):
    # Calculate the weighted sum
    z = W0 + np.sum(X * np.array([W1, W2]), axis=1)
    # Calculate the output
    output = np.array([step(val) for val in z])
    # Calculate the error
    error = y - output
    # Update the weights
    W0 += learning_rate * error.sum()
    W1 += learning_rate * np.dot(error, X[:, 0])
    W2 += learning_rate * np.dot(error, X[:, 1])
    # Increment the iteration count
    iteration_count += 1
    # If the perceptron has converged, stop training
    if np.sum(error) == 0:
      break
  # Add the number of iterations to the list
  iterations.append(iteration_count)
# Plot the number of iterations taken for learning to converge against the learning rates
plt.plot(learning_rates, iterations)
plt.xlabel('Learning rate')
plt.ylabel('Number of iterations to converge')
plt.title('the number of iterations taken for learning to converge against the learning rates')
plt.show()
#a1
import numpy as np
import matplotlib.pyplot as plt
# initial weights, learning rate, and convergence threshold
initial_weights = np.array([-10, 0.2, -0.75])
learning_rate = 0.05
convergence_threshold = 0.002
#training data for the XOR gate
training_data_xor = [
    {'input': np.array([1, 0, 0]), 'output': 0},
    {'input': np.array([1, 0, 1]), 'output': 1},
    {'input': np.array([1, 1, 0]), 'output': 1},
    {'input': np.array([1, 1, 1]), 'output': 0},
]
# Define the step activation function
def step_function(x):
    return 1 if x >= 0 else 0
# Training the perceptron
max_epochs = 1000
error_values = []
weights = np.copy(initial_weights)  # Initialize weights
for epoch in range(max_epochs):
    total_error = 0
    for obs in training_data_xor:
        input_data = obs['input']
        target_output = obs['output']
        # Calculate the weighted sum
        weighted_sum = np.dot(input_data, weights)
        # Apply the step activation function
        predicted_output = step_function(weighted_sum)
        # Calculate the error
        error = target_output - predicted_output
        # Update the weights
        delta_weights = learning_rate * error * input_data
        weights += delta_weights
        total_error += error
    error_values.append(total_error)
    # Print the total error for this epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Error = {total_error}")
    # Check for convergence
    if abs(total_error) <= convergence_threshold:
        print(f"Convergence achieved at epoch {epoch}")
        break
# plot of epochs against error values
plt.plot(range(len(error_values)), error_values)
plt.xlabel('Epochs')
plt.ylabel('Total Error')
plt.title('Epochs vs. Total Error')
plt.show()
#final weights
print("Final Weights:")
print(weights)
# Test the perceptron with XOR gate inputs
test_inputs = [
    np.array([1, 0, 0]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1]),
]
print("Test Results:")
for test_input in test_inputs:
    weighted_sum = np.dot(test_input, weights)
    output = step_function(weighted_sum)
    print(f"Input: {test_input[0:]} => Output: {output}")
 #A5
import numpy as np
# Define initial weights and learning rate
W0 = 0.3
W1 = 0.3
W2 = 0.3
W3 = 0.3
learning_rate = 0.3
# Training data
data = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])
# Target values (High Value or Low Value)
targets = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def predict(x1, x2, x3, x4):
    weighted_sum = W0 + W1 * x1 + W2 * x2 + W3 * x3
    return sigmoid(weighted_sum)
def train_perceptron(max_epochs, data, targets):
    global W0, W1, W2, W3  # Declare global variables
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2, x3, x4 = data[i]
            target = targets[i]
            prediction = predict(x1, x2, x3, x4)
            error = target - prediction
            total_error += error
            W0 += learning_rate * error
            W1 += learning_rate * error * x1
            W2 += learning_rate * error * x2
            W3 += learning_rate * error * x3
        if total_error == 0:
            break
# Train the perceptron
train_perceptron(1000, data, targets)
# Test the perceptron and print the results
for i in range(len(data)):
    x1, x2, x3, x4 = data[i]
    prediction = predict(x1, x2, x3, x4)
    classification = "Yes" if prediction >= 0.5 else "No"
    print(f"Customer C_{i + 1}: {classification}")










