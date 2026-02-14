import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dataset load
data = pd.read_csv("Housing.csv")

# Yes/No ko 1/0 me convert
data.replace({'yes':1, 'no':0}, inplace=True)

# Features aur target
X = data.drop("price", axis=1)
y = data["price"]

# Model train
model = LinearRegression()
model.fit(X, y)

# Model save
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")