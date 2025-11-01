# Method for training models
def train_models(self, models=['dt','ab','rf', 'xgb', 'svm'], task='both', hyperparameter_tuning=False, show_viz=False):
    if 'dt' in models:
        self.train_decision_tree(task, hyperparameter_tuning, show_viz)
    if 'ab' in models:
        self.train_adaboost(task, hyperparameter_tuning, show_viz)
    if 'rf' in models:
        self.train_random_forest(task, hyperparameter_tuning, show_viz)
    if 'svm' in models:
        self.train_svm(task, hyperparameter_tuning, show_viz)
    if 'xgb' in models:
        self.train_xgboost(task, hyperparameter_tuning, show_viz)
    return self