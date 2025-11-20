
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b


calc = Calculator()

x = int(input("Enter first number: "))
y = int(input("Enter second number: "))

print("Addition:", calc.add(x, y))
print("Subtraction:", calc.subtract(x, y))
print("Multiplication:", calc.multiply(x, y))
print("Division:", calc.divide(x, y))
