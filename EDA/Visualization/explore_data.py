from DataImporting.ImportData import get_data
from DataImporting.ImportData import refactor_labels
import numpy as np
from scipy import stats

file_g = r"C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeatures\data_features_for_each_matrix.xlsx"
group_column = "group"
df = refactor_labels(get_data(file_g, 'Sheet1'), group_column)
print("df.shape\n",df.shape)
df = df.dropna(axis=1)
z = np.abs(stats.zscore(df))
threshold = 3
print(np.where(z > threshold))
#
# print("names\n", list(df))
# print("count missing\n", df.isnull().sum())
# print("\ndf.info()\n",df.info())
# print("\ndf.describe()\n", df.describe())
# print("\ndf groupby count\n", df.groupby(group_column)[group_column].count())
# print("\ndf groupby describe\n", df.groupby(group_column).describe())