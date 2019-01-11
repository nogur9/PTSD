# get data
dataset = refactor_labels(get_data(path, 'Sheet1'), group_column)

# all the visualizations
# auto_visualize_features(dataset.drop([subject_number_column], 1))

# remove missing values columns
# non_missing_values_treshold = len(dataset.index) * 0.99, thresh=non_missing_values_treshold,
dataset.dropna(axis=1, inplace=True)
# set X
X = dataset.drop([group_column, subject_number_column], 1)
sbj = dataset[subject_number_column]
Y = dataset[group_column]