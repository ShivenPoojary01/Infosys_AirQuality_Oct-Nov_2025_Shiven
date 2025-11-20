# Open a file (modes: r, w, a, r+, w+, a+)
f = open("example.txt", "w")   # write mode (creates file if not exists)

# Write to file
f.write("Hello World!\n")
f.write("Python File Handling\n")
f.close()

# Read entire file
f = open("example.txt", "r")
print("Read all:\n", f.read())
f.close()

# Read line by line
f = open("example.txt", "r")
print("\nRead one line:", f.readline())   # first line
print("Read remaining lines:", f.readlines())  # list of remaining lines
f.close()

# Append to file
f = open("example.txt", "a")
f.write("Appending a new line.\n")
f.close()

# Read after append
f = open("example.txt", "r")
print("\nAfter append:\n", f.read())
f.close()

# Using with (auto closes file)
with open("example.txt", "r") as f:
    data = f.read()
    print("\nUsing with:\n", data)

# Check file pointer position
with open("example.txt", "r") as f:
    print("\nCurrent position:", f.tell())  # pointer location
    f.read(5)
    print("After reading 5 chars, position:", f.tell())
    f.seek(0)   # reset pointer
    print("After seek(0):", f.readline())
