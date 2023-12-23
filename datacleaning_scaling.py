from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([[1, 2], [3, 4], [5, 6]])

# Create a StandardScaler object
scaler = StandardScaler()

# Alternatively, use fit_transform() to both fit the scaler and transform the data
transformed_data_2 = scaler.fit_transform(data)

print("\nTransformed data using fit_transform():")
print(transformed_data_2)
