import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data (age, income, bought_product)
data = {
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 80000, 90000, 120000, 150000],
    'bought_product': [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['age', 'income']]
y = df['bought_product']

# Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
