# Dataset for Regression tasks
self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(X, y_regression, test_size=test_size, random_state=self.random_state, shuffle=True)
# Dataset for Classification tasks
self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = train_test_split(X, y_classification, test_size=test_size, random_state=self.random_state, stratify=y_classification, shuffle=True)