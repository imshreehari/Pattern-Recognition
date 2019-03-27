
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Data Cleaning
df = pd.read_csv("breast-cancer-wisconsin.txt", header=None, sep="\t")
df = df[df[6] != '?']
df[6]=df[6].astype('int64')
df.to_csv("data.txt", sep="\t", index=False, header=False)
df=df.values;

# Spilt train set and test set
X=df[:,range(1,10)]
y=df[:,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

# Model Checking
print(classification_report(y_test, y_pred))