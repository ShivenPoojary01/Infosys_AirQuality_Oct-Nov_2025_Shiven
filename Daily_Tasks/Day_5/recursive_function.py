#Factorial Calculation using Recursion
def factorial(n):
    """
    Calculates the factorial of a non-negative integer using recursion.
    """
    # Base Case: 0! = 1
    if n == 0 or n == 1:
        return 1
    
    # Recursive Step: n! = n * (n-1)!
    else:
        return n * factorial(n - 1)

# --- Example ---
num_fact = 5
print(f"The factorial of {num_fact} is: {factorial(num_fact)}") 

#Fibonacci Sequence using Recursion
def fibonacci(n):
    """
    Returns the nth Fibonacci number using recursion.
    """
    # Base Cases
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    # Recursive Step: F(n) = F(n-1) + F(n-2)
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)