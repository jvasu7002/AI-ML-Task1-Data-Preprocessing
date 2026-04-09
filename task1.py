import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/vasu jain/OneDrive/Downloads/Desktop/titanic dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column (too many missing values)
df = df.drop('Cabin', axis=1)

# Convert 'Sex' into numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked into multiple columns
df = pd.get_dummies(df, columns=['Embarked'])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Scale Age and Fare
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Show boxplot
plt.boxplot(df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# Remove outliers using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

# Keep only normal data
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

print("\nCleaned Data Shape:", df.shape)
print(df.head())
