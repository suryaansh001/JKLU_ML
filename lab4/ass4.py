import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
# Load dataset
df = pd.read_csv('profits.txt', header=None, names=['pop', 'profit'])

# Visualizing data
plt.scatter(df['pop'], df['profit'])
plt.xlabel('Population of City')
plt.ylabel('Profit')
plt.title('Population vs Profit')
plt.show()
plt.savefig('population_vs_profit.png')

# Gradient Descent Parameters
total_iter = 100
learning_rate = 0.01

def compute_cost(theta, X, Y):
    """Calculate cost function J(theta)."""
    m = len(Y)
    predictions = np.dot(X, theta)
    cost = np.sum(np.square(predictions - Y)) / (2 * m)
    return cost

def gradient_descent(X, Y, theta, learning_rate, total_iter):
    """Performs batch gradient descent."""
    m = len(Y)
    cost_history = []

    for i in range(total_iter):
        predictions = np.dot(X, theta)
        errors = predictions - Y
        theta -= (learning_rate / m) * np.dot(X.T, errors)
        cost_history.append(compute_cost(theta, X, Y))

    return theta, cost_history

# Prepare data
m = len(df)
X = np.c_[np.ones(m), df['pop']]  # Add bias term
Y = df['profit'].values  # Convert to NumPy array
theta = np.zeros(X.shape[1])  # Initialize parameters

# Run gradient descent
theta, cost_history = gradient_descent(X, Y, theta, learning_rate, total_iter)

print("Optimized Theta:", theta)
print("Final Cost:", cost_history[-1])

# Plot cost function over iterations
plt.plot(range(total_iter), cost_history, label="Cost Function")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs Iterations')
plt.legend()
plt.show()
plt.savefig('cost_vs_iterations.png')

# Prediction
user_input = float(input("Enter the population of the city: "))
predicted_profit = theta[0] + theta[1] * user_input
print(f"Predicted Profit: {predicted_profit:.2f}")

# 3D Plot of Cost vs. Parameters
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(t, X, Y)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap="viridis")
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost Function')
ax.set_title('3D Surface Plot of Cost Function')
plt.show()
plt.savefig('3d_surface_plot.png')
