import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# There is one random column available in the csv
df.drop(columns=[df.columns[0]], inplace=True)
df.drop(columns=["CustomerID"], inplace=True)

# Correcting 'Gender' column inconsistencies
df['Gender'] = df['Gender'].replace('Fe male', 'Female')

# Correcting 'MaritalStatus' column inconsistencies
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# NOTE: There are few columns which are intentionally left as raw strings.
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) ProdTaken ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
