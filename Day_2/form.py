# Form with all data types

# int
age = int(input("Enter your age (int): "))

# float
salary = float(input("Enter your salary (float): "))

# complex
z = complex(input("Enter a complex number (e.g., 2+3j): "))

# string
name = input("Enter your name (string): ")

# bool
is_student = input("Are you a student? (yes/no): ")
is_student = True if is_student.lower() == "yes" else False

# list
hobbies = input("Enter your hobbies separated by commas: ").split(",")

# tuple
marks = tuple(map(int, input("Enter 3 marks separated by spaces: ").split()))

# set
unique_numbers = set(map(int, input("Enter numbers separated by spaces (duplicates removed): ").split()))

# dictionary
details = {
    "id": int(input("Enter your ID (int): ")),
    "city": input("Enter your city: ")
}

print("\n--- Form Data ---")
print("Name:", name)
print("Age:", age)
print("Salary:", salary)
print("Complex Number:", z)
print("Student:", is_student)
print("Hobbies (List):", hobbies)
print("Marks (Tuple):", marks)
print("Unique Numbers (Set):", unique_numbers)
print("Details (Dictionary):", details)
print("Optional Field (NoneType):", optional_field)
