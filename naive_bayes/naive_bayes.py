# Homework 2 : วิชา Mathematical for Data Science ในเรื่องของ Naive-Bayes
# สามารถตรวจสอบ Homework 2 ได้จากไฟล์ "homework2.png"
from fractions import Fraction


# Step 1 : Create a tables (สร้างตารางสำหรับการเก็บข้อมูล)
features = ["Age", "Income", "Student", "Credit Rating"]


data = [
    ["young", "high", "no", "fair", "no"],
    ["young", "high", "no", "excellent", "no"],
    ["middle", "high", "no", "fair", "yes"],
    ["senior", "medium", "no", "fair", "yes"],
    ["senior", "low", "yes", "fair", "yes"],
    ["senior", "low", "yes", "excellent", "no"],
    ["middle", "low", "yes", "excellent", "yes"],
    ["young", "medium", "no", "fair", "no"],
    ["young", "low", "yes", "fair", "yes"],
    ["senior", "medium", "yes", "fair", "yes"],
    ["young", "medium", "yes", "excellent", "yes"],
    ["middle", "medium", "no", "excellent", "yes"],
    ["middle", "high", "yes", "fair", "yes"],
    ["senior", "medium", "no", "excellent", "no"],
]


# Step 2 : Define the new data X to predict (กำหนดข้อมูล X ที่ต้องการทำนาย)
query = ["middle", "high", "yes", "fair"]


# Step 3 : Prior P(Buy) = n(Buy) / n (คำนวณหาความน่าจะเป็นของแต่ละ class)
def prior(data, label):
    count = sum(1 for row in data if row[-1] == label)
    return Fraction(count, len(data)), count


# Step 4 : Likelihood P(x_i | Buy) = n(x_i, Buy) / n(Buy) (คำนวณหาความน่าจะเป็นของแต่ละ attribute เมื่อรู้ class)
def likelihood(data, index, value, label):
    rows = [row for row in data if row[-1] == label]
    count = sum(1 for row in rows if row[index] == value)
    return Fraction(count, len(rows)), count, len(rows)


# Step 5 : P(Buy | X) ∝ P(Buy) * product(P(x_i | Buy)) (คำนวณหา Posterior แล้วเลือก class ที่มีค่ามากที่สุด)
def predict(data, query):
    labels = sorted(set(row[-1] for row in data))
    scores = {}

    for label in labels:
        p_label, n_label = prior(data, label)
        print(f"P(Buy={label}) = {n_label}/{len(data)} = {float(p_label):.4f}")
        score = p_label

        for i, value in enumerate(query):
            p, count, total = likelihood(data, i, value, label)
            print(
                f"  P({features[i]}={value} | Buy={label}) = {count}/{total} = {float(p):.4f}"
            )
            score *= p

        scores[label] = score
        print(
            f"  => P(X | Buy={label}) * P(Buy={label}) = {score} = {float(score):.6f}\n"
        )

    total = sum(scores.values())
    for label, score in scores.items():
        posterior = score / total if total else 0
        print(f"P(Buy={label} | X) = {float(posterior):.4f}")

    return max(scores, key=scores.get)


# Step 6 : Show the prediction (แสดงผลการทำนาย)
if __name__ == "__main__":
    print(f"X = {dict(zip(features, query))}\n")
    result = predict(data, query)
    print(f"\nPrediction: Buy = {result}")
