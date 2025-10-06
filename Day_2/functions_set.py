# Function to create a set
def create_set():
    return set(map(int, input("Enter numbers separated by space: ").split()))

# Function to add element
def add_element(s, elem):
    s.add(elem)
    return s

# Function to remove element
def remove_element(s, elem):
    s.discard(elem)   # discard avoids error if element not found
    return s

# Function to check membership
def search_element(s, elem):
    return elem in s

# Function to clear set
def clear_set(s):
    s.clear()
    return s

# Function to union with another set
def union_sets(s1, s2):
    return s1.union(s2)

# Function to intersection with another set
def intersection_sets(s1, s2):
    return s1.intersection(s2)

# Function to difference
def difference_sets(s1, s2):
    return s1.difference(s2)

# Main Program
my_set = create_set()
print("Original Set:", my_set)

my_set = add_element(my_set, int(input("Enter number to add: ")))
print("After Adding:", my_set)

my_set = remove_element(my_set, int(input("Enter number to remove: ")))
print("After Removing:", my_set)

find = int(input("Enter number to search: "))
print("Found:", search_element(my_set, find))

set2 = create_set()
print("Second Set:", set2)
print("Union:", union_sets(my_set, set2))
print("Intersection:", intersection_sets(my_set, set2))
print("Difference:", difference_sets(my_set, set2))

print("Cleared Set:", clear_set(my_set))
