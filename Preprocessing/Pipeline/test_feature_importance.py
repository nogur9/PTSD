rfc = RandomForestClassifier()

prepro = Pipeline([
    ('missing_values', RemoveMissingFeaturesTransformer()),
    ('scaling', MinMaxScaler()),
    ('variance_threshold', VarianceThreshold()),
    ('correlation_threshold', RemoveCorrelationTransformer()),

])

('rfc', FeatureImportanceTransformer()),