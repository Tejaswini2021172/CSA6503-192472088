import pandas as pd

data = {
    "Name": ["Raj", "Ravi", "Priya"],
    "Marks": [85, 90, 88]
}

df = pd.DataFrame(data)

df["Result"] = ["Pass", "Pass", "Pass"]

print(df)
