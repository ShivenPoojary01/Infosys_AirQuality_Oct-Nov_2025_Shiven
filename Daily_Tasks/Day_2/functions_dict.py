# Create dictionary
student = {
    "name": "Alice",
    "age": 20,
    "course": "CS",
    "marks": 88
}

print("Original Dictionary:", student)

# len()
print("Length:", len(student))

# keys()
print("Keys:", student.keys())

# values()
print("Values:", student.values())

# items()
print("Items:", student.items())

# get()
print("Get age:", student.get("age"))

# update()
student.update({"marks": 92})
print("After Update:", student)

# pop()
student.pop("course")
print("After Pop:", student)

# popitem()
student.popitem()
print("After Popitem:", student)

# setdefault()
student.setdefault("city", "Mumbai")
print("After Setdefault:", student)

# copy()
student_copy = student.copy()
print("Copy of Dictionary:", student_copy)

# clear()
student.clear()
print("After Clear:", student)
