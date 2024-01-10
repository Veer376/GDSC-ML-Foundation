data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Walling ford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]
from sklearn.feature_extraction import DictVectorizer

vector = DictVectorizer(sparse=False, dtype=float)  # creation of the object of type DictVectorizer;
featured_data = vector.fit_transform(data)
feature_names = vector.get_feature_names_out()  # extraction of the faetures names from the data;
for i, data_row in enumerate(featured_data):
    print(f"Data {i + 1}:")
    for j, feature_value in enumerate(data_row):
        print(f"    {feature_names[j]}: {feature_value}")
sample = ['problem of evil',
          'evil queen',
          'horizon problem']

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# create a feature extraction object
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
print(pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out()))

# ## implementing the vector.fit_transform(data) function. #great.
# # Identify unique features
# unique_features = set()
# for d in data:
#     unique_features.update(d.keys())
#
# # Create a matrix (list of lists) filled with zeros
# transformed_data_manually = []
# for d in data:
#     row = []
#     for feature in unique_features:
#         if feature in d:
#             row.append(d[feature])
#         else:
#             row.append(0)  # You can choose a different default value if needed
#     transformed_data_manually.append(row)
#
# # Print transformed data with labels
# for i, data_row in enumerate(transformed_data_manually):
#     print(f"Data {i + 1}:")
#     for j, feature_value in enumerate(data_row):
#         feature_name = list(unique_features)[j]  # Get the feature name from the set
#         print(f"    {feature_name}: {feature_value}")
#
#
