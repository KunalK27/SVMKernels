import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('E:\VIT\Fall Semester21-22\ML\corel'):
for filename in filenames:
os.path.join(dirname, filename)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
from tqdm import tqdm
DATADIR = 'E:\VIT\Fall Semester21-22\ML\corel'
CATEGORIES = ['elep','hor']
IMG_SIZE=100
for category in CATEGORIES:
path=os.path.join(DATADIR, category)
for img in os.listdir(path):
img_array=cv2.imread(os.path.join(path,img))
plt.imshow(img_array)
plt.show()
break
break
training_data=[]
def create_training_data():
for category in CATEGORIES:
path=os.path.join(DATADIR, category)
class_num=CATEGORIES.index(category)
for img in os.listdir(path):
try:
img_array=cv2.imread(os.path.join(path,img))
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
training_data.append([new_array,class_num])
except Exception as e:
pass
create_training_data()
print(len(training_data))
lenofimage = len(training_data)
X=[]
y=[]
for categories, label in training_data:
X.append(categories)
y.append(label)
X= np.array(X).reshape(lenofimage,-1)
X.shape
X = X/255.0
X[1]
y=np.array(y)
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.svm import SVC
svc = SVC(kernel='rbf',gamma='auto')
svc.fit(X_train, y_train)
y2 = svc.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))
from sklearn.metrics import classification_report
print("Accuracy on unknown data is",classification_report(y_test,y2))
result = pd.DataFrame({'original' : y_test,'predicted' : y2})
result
