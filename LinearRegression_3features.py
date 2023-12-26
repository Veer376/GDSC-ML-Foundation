import math
import pandas as panda
from matplotlib import pyplot as plot
from sklearn.linear_model import LinearRegression
import numpy as np

data=panda.read_csv(r"C:\Users\aryav\OneDrive\Documents\houseprices_dummy_data.csv")
fillbedroom=math.floor(data.bedrooms.median())
data.bedrooms=data.bedrooms.fillna(fillbedroom)
area=np.array([data.area]).reshape(-1,1)
bedrooms=np.array([data.bedrooms]).reshape(-1,1)
age=np.array([data.age]).reshape(-1,1)
features=np.concatenate((area,bedrooms,age),axis=1)

label=np.array(data.price)
model=LinearRegression().fit(features,label)
pridiction=model.predict(features)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')

# Extracting individual dimensions
x = features[:,0]
y = features[:, 1]
z = features[:, 2]

ax.scatter(x, y, z, c=label)
ax.plot(x,y,z,pridiction)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plot.title('4D Scatter Plot')
plot.show()
print(model.predict([[3000,3,40]]))