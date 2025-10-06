# Creating two sets
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

print("Set A:", A)
print("Set B:", B)

# Union
print("Union (A ∪ B):", A | B)
print("Union using method:", A.union(B))

# Intersection
print("Intersection (A ∩ B):", A & B)
print("Intersection using method:", A.intersection(B))

# Difference
print("Difference (A - B):", A - B)
print("Difference using method:", A.difference(B))

# Symmetric Difference
print("Symmetric Difference (A Δ B):", A ^ B)
print("Symmetric Difference using method:", A.symmetric_difference(B))

# Subset and Superset
print("Is A subset of B?", A.issubset(B))
print("Is A superset of B?", A.issuperset(B))

# Disjoint
print("Are A and B disjoint?", A.isdisjoint(B))
