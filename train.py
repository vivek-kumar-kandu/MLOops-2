import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv('Data/loan_data.csv')
# print(df.head())

X=df.iloc[:, 0:4]  # Features
y=df.iloc[:, -1]    # Target variable

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42,max_depth=10)# Initialize the model

model.fit(x_train, y_train)  # Train the model

accuracy = model.score(x_test, y_test)  # Evaluate the model

#save the model to file
joblib.dump(model, 'model/loan_model.pkl')  # Save the trained model to a file

print("Model trained and saved successfully!")

