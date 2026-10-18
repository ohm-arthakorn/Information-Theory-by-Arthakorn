# Homework 1 : วิชา Mathematical for Data Science ในเรื่อง Decision Tree
# สามารถตรวจสอบ Homework 1 ได้ตามไฟล์ "homework1.png"
from collections import Counter
from math import log2

# Step 1 : Create a tables (สร้างตารางสำหรับการเก็บข้อมูล)
columns = ['age', 'income', 'student', 'credit_rating', 'buy_computer']

rows = [
    ['youth', 'high', 'no', 'fair', 'no'],
    ['youth', 'high', 'no', 'excellent', 'no'],
    ['middle_aged', 'high', 'no', 'fair', 'yes'],
    ['senior', 'medium', 'no', 'fair', 'yes'],
    ['senior', 'low', 'yes', 'fair', 'yes'],
    ['senior', 'low', 'yes', 'excellent', 'no'],
    ['middle_aged', 'low', 'yes', 'excellent', 'yes'],
    ['youth', 'medium', 'no', 'fair', 'no'],
    ['youth', 'low', 'yes', 'fair', 'yes'],
    ['senior', 'medium', 'yes', 'fair', 'yes'],
    ['youth', 'medium', 'yes', 'excellent', 'yes'],
    ['middle_aged', 'medium', 'no', 'excellent', 'yes'],
    ['middle_aged', 'high', 'yes', 'fair', 'yes'],
    ['senior', 'medium', 'no', 'excellent', 'no'],
]

data = [dict(zip(columns, row)) for row in rows]

target = 'buy_computer'
attributes = ['age', 'income', 'student', 'credit_rating']


# Step 2 : Entropy H(D) = -sum(p_i * log2(p_i)) (คำนวณหาความไม่บริสุทธิ์ของข้อมูล)
def entropy(data):
    counts = Counter(row[target] for row in data)
    total = len(data)
    return -sum((n / total) * log2(n / total) for n in counts.values())


# Step 3 : Entropy_split = sum(n_i / n * Entropy(i)) (คำนวณหา Entropy หลังจาก split ด้วย attribute)
def split_entropy(data, attribute):
    total = len(data)
    result = 0
    for value in sorted(set(row[attribute] for row in data)):
        subset = [row for row in data if row[attribute] == value]
        result += (len(subset) / total) * entropy(subset)
    return result


# Step 4 : Gain = Entropy - Entropy_split (คำนวณหา Information Gain)
def gain(data, attribute):
    return entropy(data) - split_entropy(data, attribute)


# Step 5 : Build tree with ID3 (สร้างต้นไม้โดยเลือก attribute ที่มี Gain มากที่สุด)
def id3(data, attributes, depth=0):
    labels = [row[target] for row in data]
    indent = '    ' * depth

    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return Counter(labels).most_common(1)[0][0]

    print(f"{indent}Entropy = {entropy(data):.3f}")
    gains = {attribute: gain(data, attribute) for attribute in attributes}
    for attribute, value in gains.items():
        print(f"{indent}  Entropy_{attribute} = {split_entropy(data, attribute):.3f}, Gain_{attribute} = {value:.3f}")

    best = max(gains, key=gains.get)
    print(f"{indent}  => Choose '{best}'\n")

    tree = {best: {}}
    remaining = [attribute for attribute in attributes if attribute != best]
    for value in sorted(set(row[best] for row in data)):
        subset = [row for row in data if row[best] == value]
        print(f"{indent}[{best} = {value}]")
        tree[best][value] = id3(subset, remaining, depth + 1)
    return tree


# Step 6 : Show the decision tree (แสดงผลต้นไม้ Decision Tree)
def print_tree(tree, indent=''):
    if not isinstance(tree, dict):
        print(f"{indent}--> {target} = {tree}")
        return
    for attribute, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}{attribute} = {value}")
            print_tree(subtree, indent + '    ')


if __name__ == '__main__':
    tree = id3(data, attributes)
    print('\nDecision Tree')
    print_tree(tree)
