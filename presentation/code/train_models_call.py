# Regression, without hyperparameter tuning and with hyperparameter tuning
trainer.train_models(models=['dt', 'ab', 'rf', 'xgb', 'svm'], task='regression', hyperparameter_tuning=False, show_viz=True)
trainer.train_models(models=['dt', 'ab', 'rf', 'xgb', 'svm'], task='regression', hyperparameter_tuning=True, show_viz=True)
# Print model comparison - Regression
trainer.print_model_comparison()
# Classification, without hyperparameter tuning and with hyperparameter tuning
trainer.train_models(models=['dt', 'ab', 'rf', 'xgb', 'svm'], task='classification', hyperparameter_tuning=False, show_viz=True)
trainer.train_models(models=['dt', 'ab', 'rf', 'xgb', 'svm'], task='classification', hyperparameter_tuning=True, show_viz=True)
# Print model comparison - Classification
trainer.print_model_comparison()
