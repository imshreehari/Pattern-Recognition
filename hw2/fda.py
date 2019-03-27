import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Sort
X_train_good = X_train_std[y_train==1]
X_train_bad = X_train_std[y_train==0]

# Calculate the mean vector
mean1 = np.mean(X_train_good, axis=0)
mean0 = np.mean(X_train_bad,axis=0)

# Calculate SS within classes

SS_1=0

for i in range(X_train_good.shape[0]):
    x=X_train_good[i,:]-mean1
    SS_1 += np.dot(x.reshape(9,1),x.reshape(1,9))


SS_2=0

for i in range(X_train_bad.shape[0]):
    x = X_train_bad[i, :] - mean0
    SS_1 += np.dot(x.reshape(9, 1), x.reshape(1, 9))


SS_within=SS_1+SS_2

w= np.linalg.inv(SS_within).dot(mean1-mean0)

w0 = w.dot(mean0+mean1)/2
#w0 = w.dot(X_train_bad.shape[0]*mean0+X_train_good.shape[0]*mean1)/(X_train_bad.shape[0]+X_train_good.shape[0])

y_pred=np.zeros(X_test_std.shape[0])

for i in range(X_test_std.shape[0]):
    x= X_test_std[i,:]
    if ( np.dot(x,w) > w0 ):
        y_pred[i] = 1
    else:y_pred[i] = 0

print(classification_report(y_test, y_pred))