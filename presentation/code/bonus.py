trainer = SupervisedLearning(drop_columns=[... list of the columns you want to drop ...])

dataset_file = 'synthetic_ghg_data.csv'

trainer.load_data(
    filepath=dataset_file,
    scope_col='scope1+2total',
    revenue_col='revenuesghgco',
    realzero_col='realzero',
    normalize_columns=True
)