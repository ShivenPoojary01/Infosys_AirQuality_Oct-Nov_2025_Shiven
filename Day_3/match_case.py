def calculate(a, b, operation):
    match (operation, a, b):
        
        # Case 1: Addition
        case ('add', x, y):
            # This case matches the string 'add' and binds the numbers to x and y
            return x + y
        
        # Case 2: Subtraction
        case ('subtract', x, y):
            return x - y
            
        # Case 3: Multiplication
        case ('multiply', x, y):
            return x * y
            
        # Case 4: Division, with a guard (if b != 0) to prevent ZeroDivisionError
        case ('divide', x, y) if y != 0:
            return x / y
            
        # Case 5: Division by Zero Error
        case ('divide', _, 0):
            # The underscore (_) means we don't care about the value of 'a' in this specific case
            return "Error: Cannot divide by zero!"
            
        # Default Case: Handle any unknown operation
        case _:
            return f"Error: Unknown operation '{operation}'"

# --- Testing the Calculator ---

# Addition
result_add = calculate(10, 5, 'add')
print(f"10 + 5 = {result_add}")

# Subtraction
result_sub = calculate(10, 5, 'subtract')
print(f"10 - 5 = {result_sub}")

# Multiplication
result_mul = calculate(10, 5, 'multiply')
print(f"10 * 5 = {result_mul}")

# Division
result_div = calculate(10, 5, 'divide')
print(f"10 / 5 = {result_div}")

# Division by Zero (Testing the guard/error case)
result_zero_div = calculate(10, 0, 'divide')
print(f"10 / 0 = {result_zero_div}")

# Unknown Operation (Testing the default case)
result_unknown = calculate(10, 5, 'power')
print(f"10 power 5 = {result_unknown}")