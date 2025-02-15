import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


# Function to get valid integer input
def get_integer_input(prompt, default_value, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Please enter a value greater than or equal to {min_value}.")
            elif max_value is not None and value > max_value:
                print(f"Please enter a value less than or equal to {max_value}.")
            else:
                return value
        except ValueError:
            print(f"Invalid input. Using default value {default_value}.")
            return default_value

# Get user input for the random walk parameters
steps_1d = get_integer_input("Enter the number of steps for 1D walk (default 1000): ", 1000, 1)
steps_2d = get_integer_input("Enter the number of steps for 2D walk (default 500): ", 500, 1)
steps_3d = get_integer_input("Enter the number of steps for 3D walk (default 500): ", 500, 1)
biased_prob = get_integer_input("Enter the probability for moving right in biased walk (default 70 for 70%): ", 70, 0, 100)
num_walkers = get_integer_input("Enter the number of walkers for comparison (default 5): ", 5, 1)

# 1D Random walk
positions = np.zeros(steps_1d)  
steps_array = np.random.choice([-1, 1], size=steps_1d)  
positions[1:] = np.cumsum(steps_array)[:-1]  

plt.figure(figsize=(10, 5))
plt.plot(positions, label="1D Random Walk", color="blue")
plt.xlabel("Steps")
plt.ylabel("Position")
plt.title(f"1D Random Walk Simulation (Steps: {steps_1d})")
plt.legend()
plt.grid()
plt.show(block=True)
plt.close()




# 2D Random Walk
steps_2d = 500

# Initialize position arrays
x = np.zeros(steps_2d)
y = np.zeros(steps_2d)

# Generate random movement in x and y directions
dx = np.random.choice([-1, 1], size=steps_2d)
dy = np.random.choice([-1, 1], size=steps_2d)

# Compute positions
x[1:] = np.cumsum(dx)[:-1]
y[1:] = np.cumsum(dy)[:-1]

# Create figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(min(x) - 2, max(x) + 2)
ax.set_ylim(min(y) - 2, max(y) + 2)
ax.set_xlabel("Steps")
ax.set_ylabel("Position")
ax.set_title("2D Random Walk Animation")

# Plot elements
line, = ax.plot([], [], "g-", alpha=0.7)
start_point = ax.scatter(0, 0, color="red", label="Start", s=100)
end_point = ax.scatter([], [], color="blue", label="End", s=100)
point = ax.scatter([], [], color="black", s=50)

# Update function for animation
def update(i):
    line.set_data(x[:i], y[:i])  # Update line
    point.set_offsets([x[i], y[i]])  # Move current point
    if i == steps_2d - 1:  
        end_point.set_offsets([x[i], y[i]])  # Mark end point
    return line, point,end_point

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)
plt.legend()
plt.show()






# 3D Random Walk
steps_3d = 500

x = np.zeros(steps_3d)
y = np.zeros(steps_3d)
z = np.zeros(steps_3d)

dx = np.random.choice([-1, 1], size=steps_3d)
dy = np.random.choice([-1, 1], size=steps_3d)
dz = np.random.choice([-1, 1], size=steps_3d)

x[1:] = np.cumsum(dx)[:-1]
y[1:] = np.cumsum(dy)[:-1]
z[1:] = np.cumsum(dz)[:-1]

# Creating a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color="purple", alpha=0.7)
ax.scatter(0, 0, 0, color="red", label="Start", s=100)  # Start point
ax.scatter(x[-1], y[-1], z[-1], color="blue", label="End", s=100)  # End point

ax.set_xlabel("Horizontal Movement")
ax.set_ylabel("Vertical Movement")
ax.set_zlabel("Depth")
ax.set_title(f"3D Random Walk Simulation (Steps: {steps_3d})")
ax.legend()
plt.show(block=True)
plt.close()




# Biased Walk
probabilities = [biased_prob / 100, (100 - biased_prob) / 100]
biased_steps = np.random.choice([-1, 1], size=steps_2d, p=probabilities)
biased_positions = np.cumsum(biased_steps)

plt.figure(figsize=(10, 5))
plt.plot(biased_positions, label="Biased Random Walk", color="red")
plt.xlabel("Steps")
plt.ylabel("Position")
plt.title(f"Biased 1D Random Walk (More Right Moves, {biased_prob}% Right)")
plt.legend()
plt.grid()
plt.show()



# Multiple Walkers
plt.figure(figsize=(8, 6))
for i in range(num_walkers):
    walker_x = np.cumsum(np.random.choice([-1, 1], size=steps_2d))
    plt.plot(walker_x, label=f"Walker {i+1}")

plt.xlabel("Steps")
plt.ylabel("Position")
plt.title(f"Comparison of {num_walkers} Random Walks")
plt.legend()
plt.grid()
plt.show()



# Predict Future Steps Using Linear Regression
positions = np.cumsum(np.random.choice([-1, 1], size=steps_1d))

x = np.arange(steps_1d).reshape(-1, 1)
y = positions.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

future_steps = np.arange(steps_1d, steps_1d + 10).reshape(-1, 1)
predictions = model.predict(future_steps)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(steps_1d), positions, label="Actual Walk", color="blue")
plt.plot([steps_1d - 1, steps_1d], [positions[-1], predictions[0][0]], "blue")
plt.plot(np.arange(steps_1d, steps_1d + 10), predictions, "r--", label="Predicted Path", linewidth=2)
plt.scatter(steps_1d, predictions[0][0], color="red", marker="o", label="First Prediction")
plt.axvline(x=steps_1d, color="gray", linestyle="dotted")

plt.xlabel("Steps")
plt.ylabel("Position")
plt.title(f"Predicting Random Walk Future Path (Using Linear Regression, Steps: {steps_1d})")
plt.legend()
plt.grid()
plt.show()
