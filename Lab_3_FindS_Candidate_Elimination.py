# LAB: Find-S and Candidate Elimination Algorithms

import pandas as pd
import numpy as np

# Keep enjoy.csv in the same folder as this script.
data = pd.read_csv("enjoy.csv")

print("First five rows:")
print(data.head())
print("\nLast five rows:")
print(data.tail())
print("\nDescriptive summary:")
print(data.describe(include="all"))
print("\nDataset information:")
data.info()
print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

# Features and target
X = data.iloc[:, :-1].astype(str).values
y = data.iloc[:, -1].astype(str).values

# Find-S Algorithm
def find_s(X, y):
    hypothesis = None
    print("\nFIND-S ALGORITHM")
    print("Initial hypothesis:", hypothesis)

    for i in range(len(X)):
        if y[i].strip().lower() == "yes":
            if hypothesis is None:
                hypothesis = X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != X[i][j]:
                        hypothesis[j] = "?"
            print(f"After training example {i + 1}: {hypothesis}")

    return hypothesis

final_hypothesis = find_s(X, y)
print("Final hypothesis:", final_hypothesis)

# Candidate Elimination Algorithm
# Each hypothesis is a tuple/list of attribute constraints; '?' means any value.
def covers(hypothesis, example):
    return all(h == "?" or h == x for h, x in zip(hypothesis, example))

def more_general_or_equal(h1, h2):
    return all(a == "?" or a == b for a, b in zip(h1, h2))

def minimal_generalizations(h, example):
    result = list(h)
    for i, value in enumerate(example):
        if result[i] == "Ø":
            result[i] = value
        elif result[i] != value:
            result[i] = "?"
    return result

def minimal_specializations(h, example, domains, specific):
    result = []
    for i, value in enumerate(h):
        if value == "?":
            for domain_value in domains[i]:
                if domain_value != example[i]:
                    candidate = list(h)
                    candidate[i] = domain_value
                    if more_general_or_equal(candidate, specific):
                        result.append(candidate)
    return result

def candidate_elimination(X, y):
    n_features = X.shape[1]
    domains = [sorted(set(X[:, i])) for i in range(n_features)]
    S = [["Ø"] * n_features]
    G = [["?"] * n_features]

    print("\nCANDIDATE ELIMINATION ALGORITHM")
    print("Initial S:", S)
    print("Initial G:", G)

    for i, (example, label) in enumerate(zip(X, y)):
        label = label.strip().lower()
        if label == "yes":
            # Remove general hypotheses that do not cover this positive example.
            G = [g for g in G if covers(g, example)]
            # Generalize S minimally to cover the positive example.
            new_S = []
            for s in S:
                if covers(s, example):
                    new_S.append(s)
                else:
                    new_S.append(minimal_generalizations(s, example))
            S = [s for s in new_S if any(more_general_or_equal(g, s) for g in G)]
            # Keep only maximally specific hypotheses in S.
            S = [s for s in S if not any(s != other and more_general_or_equal(s, other) for other in S)]
        elif label == "no":
            # Specialize G minimally to exclude this negative example.
            new_G = []
            for g in G:
                if covers(g, example):
                    new_G.extend(minimal_specializations(g, example, domains, S[0]))
                else:
                    new_G.append(g)
            G = [g for g in new_G if any(more_general_or_equal(g, s) for s in S)]
            # Keep only maximally general hypotheses in G.
            G = [g for g in G if not any(g != other and more_general_or_equal(other, g) for other in G)]

        print(f"Example {i + 1} ({label}):")
        print("  Specific boundary S:", S)
        print("  General boundary G:", G)

    return S, G

S, G = candidate_elimination(X, y)
print("\nFinal Specific Boundary S:", S)
print("Final General Boundary G:", G)
