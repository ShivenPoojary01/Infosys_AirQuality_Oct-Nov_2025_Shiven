# Used AI to Know about these Functions 
import os
import math
import random

# --- 1. The 'os' Module (Operating System Interactions) ---
# The 'os' module provides a way of interacting with the operating system, 
# like managing files, directories, and getting system information.

print("--- 1. Using the 'os' Module ---")

# Get the current working directory (where the script is being executed)
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}")

# List all files and directories in the current path
# file_list = os.listdir(current_dir)
# print(f"Files and directories in this path (first 5): {file_list[:5]}")

# Check if a path exists (we check if the current directory path exists)
is_path_real = os.path.exists(current_dir)
print(f"Does the current directory path exist? {is_path_real}")

# Create a new directory (folder) safely
new_dir_name = "test_data_folder"
try:
    # os.makedirs creates the directory, exist_ok=True prevents an error if it already exists
    os.makedirs(new_dir_name, exist_ok=True)
    print(f"Successfully ensured directory '{new_dir_name}' exists.")
except OSError as e:
    # Handle permissions errors or other OS issues
    print(f"Error creating directory: {e}")
finally:
    # Clean up: remove the directory we created (if it's empty)
    # Use os.rmdir() only if the directory is empty
    # os.rmdir(new_dir_name) 
    pass # Leaving it commented out for easy running

print("-" * 30)

# --- 2. The 'math' Module (Mathematical Functions) ---
# The 'math' module provides access to common mathematical functions and constants.

print("--- 2. Using the 'math' Module ---")

# Calculate the square root of a number
num_sqrt = 64
root = math.sqrt(num_sqrt)
print(f"The square root of {num_sqrt} is: {root}")

# Calculate the sine of an angle (in radians)
angle = math.pi / 2 # math.pi is the constant 3.14159...
sine_value = math.sin(angle)
print(f"The sine of pi/2 radians is: {sine_value}") # Should be 1.0

# Calculate the ceiling (the smallest integer greater than or equal to x)
num_ceil = 5.23
ceiling = math.ceil(num_ceil)
print(f"The ceiling of {num_ceil} is: {ceiling}")

# Calculate the power (e.g., 2 raised to the power of 3)
power_result = math.pow(2, 3) # Equivalent to 2 ** 3
print(f"2 raised to the power of 3 is: {power_result}")

print("-" * 30)

# --- 3. The 'random' Module (Random Number Generation) ---
# The 'random' module is used to generate pseudo-random numbers, 
# often used in simulations, games, or cryptographic applications.

print("--- 3. Using the 'random' Module ---")

# Generate a random floating-point number between 0.0 (inclusive) and 1.0 (exclusive)
rand_float = random.random()
print(f"Random float between 0.0 and 1.0: {rand_float}")

# Generate a random integer in a specified range (inclusive start and end)
rand_int = random.randint(1, 10)
print(f"Random integer between 1 and 10 (inclusive): {rand_int}")

# Select a random element from a non-empty sequence (like a list)
my_list = ['apple', 'banana', 'cherry', 'date']
random_choice = random.choice(my_list)
print(f"Randomly chosen fruit: {random_choice}")

# Shuffle a list in place (modifies the original list)
random.shuffle(my_list)
print(f"Shuffled list: {my_list}")

# Select a specified number of unique random elements from a population
random_sample = random.sample(my_list, k=2)
print(f"Random sample of 2 unique fruits: {random_sample}")
