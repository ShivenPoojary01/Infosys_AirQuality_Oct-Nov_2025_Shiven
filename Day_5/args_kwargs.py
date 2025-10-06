# Used AI to study This topic
import math

# --- 1. Comprehensions (Creating sequences concisely) ---
print("--- 1. List, Tuple, and Dictionary Comprehensions ---")

# Data to use for transformations
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']

# 1.1. List Comprehension (Creates a list)
# Syntax: [expression for item in iterable if condition]
# Goal: Create a list of squares for all even numbers.
squares_of_evens = [n * n for n in numbers if n % 2 == 0]
print(f"List Comprehension (Squares of evens): {squares_of_evens}")
# Result: [4, 16, 36, 64, 100]

# 1.2. Set Comprehension (Creates a set, automatically removing duplicates)
# Goal: Create a set of the first letter of each fruit.
first_letters = {fruit[0] for fruit in fruits}
print(f"Set Comprehension (First letters): {first_letters}")
# Result: {'c', 'e', 'a', 'b', 'd'}

# 1.3. Dictionary Comprehension (Creates a dictionary)
# Syntax: {key_expression: value_expression for item in iterable}
# Goal: Create a dictionary where the key is the number and the value is its cube.
cubes_dict = {n: n**3 for n in numbers if n < 6}
print(f"Dictionary Comprehension (Number: Cube): {cubes_dict}")
# Result: {1: 1, 2: 8, 3: 27, 4: 64, 5: 125}

# 1.4. Tuple Comprehension (Creates a Generator Expression, not a tuple object)
# Note: Using () creates a generator, which is memory efficient. 
# To get a tuple, we must explicitly wrap the generator in tuple().
generator_expression = (math.log(n) for n in numbers)
# print(f"Tuple Comprehension (Generator): {generator_expression}") # Prints the generator object

# To see the actual values (and convert to a tuple):
log_tuple = tuple(generator_expression)
print(f"Tuple from Generator (Logarithms): {log_tuple[:3]}...") 
# Result: A tuple of log values

print("-" * 50)

# --- 2. Runtime Arguments (*args and **kwargs) ---
# *args allows a function to accept an arbitrary number of positional arguments.
# **kwargs allows a function to accept an arbitrary number of keyword arguments (like a dictionary).

# 2.1. Example with *args (Positional Arguments)
def sum_all_numbers(*args):
    """Sums all positional arguments passed to the function."""
    total = 0
    # args is treated as a tuple of all passed positional arguments
    for num in args:
        total += num
    return total

print("2.1. *args Example:")
result_args = sum_all_numbers(10, 20, 30, 40, 50)
print(f"Sum of (10, 20, 30, 40, 50): {result_args}") 
result_args_short = sum_all_numbers(5, 7)
print(f"Sum of (5, 7): {result_args_short}")

# 2.2. Example with **kwargs (Keyword Arguments)
def describe_person(**kwargs):
    """Prints a description based on keyword arguments."""
    print("\n2.2. **kwargs Example:")
    # kwargs is treated as a dictionary of all passed keyword arguments
    if 'name' in kwargs:
        print(f"Name: {kwargs['name']}")
    if 'age' in kwargs:
        print(f"Age: {kwargs['age']}")
    if 'city' in kwargs:
        print(f"City: {kwargs['city']}")
    
    # Print any extra arguments passed
    for key, value in kwargs.items():
        if key not in ['name', 'age', 'city']:
            print(f"Extra Detail: {key.capitalize()}: {value}")

describe_person(name="Alice", age=25, city="New York", occupation="Software Engineer")

# 2.3. Example combining both *args and **kwargs
def profile_creator(user_id, *titles, **details):
    """
    Demonstrates using required args, *args, and **kwargs together.
    - user_id: Required positional argument
    - *titles: A tuple of professional titles
    - **details: A dictionary of key-value details
    """
    print("\n2.3. Combined *args and **kwargs Example:")
    print(f"User ID: {user_id}")
    print(f"Titles (using *args): {', '.join(titles)}")
    
    print("Details (using **kwargs):")
    for key, value in details.items():
        print(f" - {key.capitalize()}: {value}")

profile_creator(
    101, 
    "Developer", "Mentor", "Architect", # -> *titles
    email="alice@work.com", 
    level="Senior", 
    started_year=2018 # -> **details
)
