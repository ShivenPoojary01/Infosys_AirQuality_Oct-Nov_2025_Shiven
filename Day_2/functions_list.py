# Function to create a list
def create_list():
    return list(map(int, input("Enter numbers separated by space: ").split()))

# Function to add element
def add_element(lst, elem):
    lst.append(elem)
    return lst

# Function to remove element
def remove_element(lst, elem):
    if elem in lst:
        lst.remove(elem)
    return lst

# Function to search element
def search_element(lst, elem):
    return elem in lst

# Function to sort list
def sort_list(lst):
    return sorted(lst)

# Function to reverse list
def reverse_list(lst):
    return lst[::-1]

# Main Program
numbers = create_list()
print("Original List:", numbers)

numbers = add_element(numbers, int(input("Enter number to add: ")))
print("After Adding:", numbers)

numbers = remove_element(numbers, int(input("Enter number to remove: ")))
print("After Removing:", numbers)

find = int(input("Enter number to search: "))
print("Found:", search_element(numbers, find))

print("Sorted List:", sort_list(numbers))
print("Reversed List:", reverse_list(numbers))
