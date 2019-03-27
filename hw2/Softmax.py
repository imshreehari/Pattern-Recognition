from skimage import io
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


n=10
imgCnt=0
data=np.zeros([3000,2305])

nums = np.random.choice(10,n)
for i in nums:
    imglist=glob.glob("./Pictures/"+i.astype('str')+"/*.png")
    for imgpath in imglist:
        img = io.imread(imgpath, as_gray=True)
        data[imgCnt,range(2304)] =img.reshape(2304);
        data[imgCnt,2304] = i;
        imgCnt +=1

data=data[range(imgCnt),:]

X=data[:,range(2304)]
y=data[:,2304]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Softmax Regression
lr = LogisticRegression(solver='newton-cg',multi_class='multinomial')
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

# Model Checking
print(classification_report(y_test, y_pred))