import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve,
    precision_score, recall_score
)
from scipy import stats
from sklearn.preprocessing import LabelEncoder

class SupervisedLearning:
    ##
    ## Constructor
    ##
    def __init__(self, 
                 percentiles={'low': 20, 'medium_low': 40, 'medium': 60, 'medium_high': 80},

                 dt_params={'max_depth': [3, 5, 10, 15, None], 'min_samples_split': [2, 5, 10, 20],
                           'min_samples_leaf': [1, 2, 4, 8], 'max_features': ['sqrt', 'log2', None],
                           'class_weight': [None, 'balanced']},
                 ab_params={'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3],
                           'loss': ['linear', 'square', 'exponential'], 'random_state': [42]},
                 rf_params={'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20, None],
                           'min_samples_split': [5, 10, 20], 'min_samples_leaf': [1, 2, 4],
                           'bootstrap': [True, False], 'max_features': ['sqrt', 'log2', None],
                           'class_weight': [None, 'balanced', 'balanced_subsample']},
                 svm_params={'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01],
                            'kernel': ['rbf', 'linear'], 'class_weight': [None, 'balanced']},
                 xgb_params={'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9],
                            'learning_rate': [0.01, 0.1, 0.3], 'subsample': [0.8, 1.0],
                            'colsample_bytree': [0.8, 1.0]},
                 cv_folds=5, random_state=42, n_jobs=-1, verbose=1,
                 drop_columns=[],
                 penalize_exclusions=False
                 ):
        self.percentiles = percentiles
        self.rf_params = rf_params
        self.svm_params = svm_params
        self.xgb_params = xgb_params
        self.dt_params = dt_params
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.ab_params = ab_params
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.df = None
        self.scope_col = None
        self.revenue_col = None
        self.realzero_col = None
        self.exclusions_col = None
        self.penalize_exclusions = penalize_exclusions
        self.feature_cols = None
        self.max_scope = None
        self.drop_columns = drop_columns
        self.X_train_reg = self.X_test_reg = self.y_train_reg = self.y_test_reg = None
        self.X_train_clf = self.X_test_clf = self.y_train_clf = self.y_test_clf = None
        self.X_train_reg_scaled = self.X_test_reg_scaled = None
        self.X_train_clf_scaled = self.X_test_clf_scaled = None
        self.scaler_reg = self.scaler_clf = None
        self.models = {'regression': {}, 'classification': {}}
        self.metrics = {'regression': {}, 'classification': {}}
        self.best_reg_name = self.best_clf_name = None
        self.best_reg_name_untuned = self.best_clf_name_untuned = None
    

    ##
    ## Load data, and apply column noarmalizations 
    ##
    def load_data(self, filepath, scope_col, revenue_col, realzero_col, exclusions_col=None, normalize_columns=True):
        try:
            self.df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(filepath, encoding='latin-1')
            except UnicodeDecodeError:
                self.df = pd.read_csv(filepath, encoding='cp1252')
        if normalize_columns:
            self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        self.scope_col = scope_col
        self.revenue_col = revenue_col
        self.realzero_col = realzero_col
        self.exclusions_col = exclusions_col
        self.df[scope_col] = pd.to_numeric(self.df[scope_col], errors='coerce')
        self.df[revenue_col] = pd.to_numeric(self.df[revenue_col], errors='coerce')
        if realzero_col:
            self.df[realzero_col] = pd.to_numeric(self.df[realzero_col], errors='coerce')
        print(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self
    
    ##
    ## Wrapper to describe the dataset before and after transformations
    ##
    def describe_data(self):
        print("Raw dataset Info:")
        print("="*80)
        self.df.info()
        print("" + "="*80)
        print("Basic Statistics:")
        print(self.df.describe())

        self.df.drop(columns=self.drop_columns, inplace=True)

        print("Final dataset Info:")
        print("="*80)
        self.df.info()
        print("" + "="*80)
        print("Basic Statistics:")
        print(self.df.describe())
        return self
    
    ##
    ## Calculating the target variables - Raw Score
    ##
    def _calc_esg_score_row(self, row):
        scope_value = row[self.scope_col]
        revenue_value = row[self.revenue_col]
        is_realzero = row[self.realzero_col] if self.realzero_col else False
        
        # Check exclusions if penalization is enabled
        has_exclusion = False
        if self.penalize_exclusions and self.exclusions_col:
            exclusion_value = row[self.exclusions_col]
            # Penalize if exclusions is 'yes' (case-insensitive) or NaN
            has_exclusion = pd.isna(exclusion_value) or (isinstance(exclusion_value, str) and exclusion_value.lower() == 'yes')
        
        if pd.isna(revenue_value) or revenue_value <= 0:
            return np.nan
        
        if pd.notna(scope_value) and scope_value >= 0:
            base_score = scope_value / revenue_value
        elif not is_realzero:
            base_score = (self.max_scope + 1) / (revenue_value * 0.75)
        else:
            # To prevent zeros, take the brands count in consideration
            if row["brands"] >= 7:
                base_score = row["brands"] / 7
            else:
                base_score = 7
        
        # Apply penalty multiplier if exclusions found
        if has_exclusion:
            if base_score == 0:
                base_score = (1 + base_score * 1.1) # Should score at least some percentage above the 50%
            else:
                base_score = 1 + base_score * 1.1
            #print(f"Has Exclusions, raw base score: {base_score}")
        
        return base_score
    
    ##
    ## Calculating the target variables - Final Score
    ##
    def calculate_esg_score(self):
        self.max_scope = pd.to_numeric(self.df[self.scope_col], errors='coerce').max()
        self.df['esg_score_raw'] = self.df.apply(self._calc_esg_score_row, axis=1)
        min_score = self.df['esg_score_raw'].min()
        mean_score = self.df['esg_score_raw'].mean()
        max_score = self.df['esg_score_raw'].max()
        self.df['esg_score'] = 100 - (((self.df['esg_score_raw'] - min_score) / (max_score - min_score)) * 100)

        #print(f"Final Score: {self.df['esg_score']}")
        return self
    
    ##
    ## Calculating the target variables - Risk based on the percents
    ##
    def _classify_risk_row(self, row):
        score = row['esg_score_raw']
        scope_value = row[self.scope_col]
        is_realzero = row[self.realzero_col] if self.realzero_col else False

        esg_risk = 'LOW'

        if (pd.isna(scope_value) and not is_realzero):
            esg_risk = 'HIGH'
            print(f"score {score} is null or realzero = {esg_risk}")
        elif score >= self.percentiles['medium_high']:
            esg_risk = 'HIGH'
            print(f"score {score} >= {self.percentiles['medium_high']} = {esg_risk}")
        elif score >= self.percentiles['medium']:
            esg_risk = 'MEDIUM-HIGH'
            print(f"score {score} >= {self.percentiles['medium']} = {esg_risk}")
        elif score >= self.percentiles['medium_low']:
            esg_risk = 'MEDIUM'
            print(f"score {score} >= {self.percentiles['medium_low']} = {esg_risk}")
        elif score >= self.percentiles['low']:
            esg_risk = 'MEDIUM-LOW'
            print(f"score {score} >= {self.percentiles['low']} = {esg_risk}")
        else:
            esg_risk = 'LOW'
            print(f"score {score} else... = {esg_risk}")
        
        
        return esg_risk
    
    ##
    ## Calculating the target variables - esg_risk - Apply
    ##
    def classify_risk(self):
        self.df['esg_risk'] = self.df.apply(self._classify_risk_row, axis=1)
        return self
    
    ##
    ## Helper to prepare the feature columns
    ##
    def prepare_features(self, feature_cols):
        self.feature_cols = feature_cols
        return self
    
    ##
    ## Wrapper to split train and test data
    ##
    def split_data(self, test_size=0.2):
        df_clean = self.df.dropna(subset=['esg_score', 'esg_risk'])
        X = df_clean[self.feature_cols]
        valid_cols = [col for col in self.feature_cols if X[col].notna().sum() > 0]
        removed_cols = [col for col in self.feature_cols if col not in valid_cols]

        if removed_cols:
            print(f"Removing {len(removed_cols)} columns with all NaN values: {removed_cols}")

        X = X[valid_cols]
        self.feature_cols = valid_cols
        y_regression = df_clean['esg_score']
        y_classification = df_clean['esg_risk']
        
        # Splitting train and test for regression tasks
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(X, y_regression, test_size=test_size, random_state=self.random_state, shuffle=True)

        # Splitting train and test for classification tasks
        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = train_test_split(X, y_classification, test_size=test_size, random_state=self.random_state, stratify=y_classification, shuffle=True)
        
        # Apply imputations
        self.imputer = SimpleImputer(strategy='median')
        self.X_train_reg = pd.DataFrame(self.imputer.fit_transform(self.X_train_reg), columns=valid_cols, index=self.X_train_reg.index)
        self.X_test_reg = pd.DataFrame(self.imputer.transform(self.X_test_reg), columns=valid_cols, index=self.X_test_reg.index)
        self.X_train_clf = pd.DataFrame(self.imputer.transform(self.X_train_clf), columns=valid_cols, index=self.X_train_clf.index)
        self.X_test_clf = pd.DataFrame(self.imputer.transform(self.X_test_clf), columns=valid_cols, index=self.X_test_clf.index)
        
        # Apply scaling for regression tasks
        self.scaler_reg = StandardScaler()
        self.X_train_reg_scaled = self.scaler_reg.fit_transform(self.X_train_reg)
        self.X_test_reg_scaled = self.scaler_reg.transform(self.X_test_reg)

        # Apply scaling for regression tasks
        self.scaler_clf = StandardScaler()
        self.X_train_clf_scaled = self.scaler_clf.fit_transform(self.X_train_clf)
        self.X_test_clf_scaled = self.scaler_clf.transform(self.X_test_clf)

        print(f"Applied median imputation to handle missing values in {len(valid_cols)} features")
        return self

    ##
    ## Helper to get the model key, for the visualizations and logs
    ##
    def _get_model_key(self, base_name, hyperparameter_tuning):
        """Generate model key with tuning suffix"""
        return f"{base_name} (Tuned)" if hyperparameter_tuning else f"{base_name} (Untuned)"
    
    ##
    ## Training Decision Tree
    ##
    def train_decision_tree(self, task='both', hyperparameter_tuning=False, show_viz=False):
        model_key = self._get_model_key('Decision Tree', hyperparameter_tuning)
        
        if task in ['regression', 'both']:
            print(f"Training {model_key} Regressor...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                param_grid = {k: v for k, v in self.dt_params.items() if k != 'class_weight'}
                grid = GridSearchCV(DecisionTreeRegressor(random_state=self.random_state),
                                  param_grid, cv=self.cv_folds, scoring='r2', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_reg, self.y_train_reg)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = DecisionTreeRegressor(random_state=self.random_state)
                best_model.fit(self.X_train_reg, self.y_train_reg)
                best_params = {}

            y_pred = best_model.predict(self.X_test_reg)
            self.models['regression'][model_key] = best_model
            self.metrics['regression'][model_key] = {
                'r2': r2_score(self.y_test_reg, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'mae': mean_absolute_error(self.y_test_reg, y_pred),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }

        if show_viz:
            self.display_visualizations(model_name=model_key, task='regression')

        if task in ['classification', 'both']:
            print(f"Training {model_key} Classifier...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                grid = GridSearchCV(DecisionTreeClassifier(random_state=self.random_state),
                                  self.dt_params, cv=self.cv_folds, scoring='accuracy', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_clf, self.y_train_clf)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = DecisionTreeClassifier(random_state=self.random_state)
                best_model.fit(self.X_train_clf, self.y_train_clf)
                best_params = {}

            y_pred = best_model.predict(self.X_test_clf)
            self.models['classification'][model_key] = best_model
            self.metrics['classification'][model_key] = {
                'accuracy': accuracy_score(self.y_test_clf, y_pred),
                'f1': f1_score(self.y_test_clf, y_pred, average='weighted'),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }
            
            if show_viz:
                self._visualize_after_training(model_key, 'classification')
        return self

    ##
    ## Training AdaBoost
    ##
    def train_adaboost(self, task='both', hyperparameter_tuning=False, show_viz=False):
        model_key = self._get_model_key('AdaBoost', hyperparameter_tuning)
        
        if task in ['regression', 'both']:
            print(f"Training {model_key} Regressor...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                param_grid = self.ab_params
                grid = GridSearchCV(AdaBoostRegressor(random_state=self.random_state),
                                    param_grid, cv=self.cv_folds, scoring='r2', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_reg, self.y_train_reg)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = AdaBoostRegressor(random_state=self.random_state)
                best_model.fit(self.X_train_reg, self.y_train_reg)
                best_params = {}

            y_pred = best_model.predict(self.X_test_reg)
            self.models['regression'][model_key] = best_model
            self.metrics['regression'][model_key] = {
                'r2': r2_score(self.y_test_reg, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'mae': mean_absolute_error(self.y_test_reg, y_pred),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }

        if show_viz:
            self._visualize_after_training(model_key, 'regression')

        if task in ['classification', 'both']:
            print(f"Training {model_key} Classifier...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                param_grid = {k: v for k, v in self.ab_params.items() if k != 'loss'}
                grid = GridSearchCV(AdaBoostClassifier(random_state=self.random_state),
                                  param_grid, cv=self.cv_folds, scoring='accuracy', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_clf, self.y_train_clf)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = AdaBoostClassifier(random_state=self.random_state)
                best_model.fit(self.X_train_clf, self.y_train_clf)
                best_params = {}

            y_pred = best_model.predict(self.X_test_clf)
            self.models['classification'][model_key] = best_model
            self.metrics['classification'][model_key] = {
                'accuracy': accuracy_score(self.y_test_clf, y_pred),
                'f1': f1_score(self.y_test_clf, y_pred, average='weighted'),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }
            if show_viz:
                self._visualize_after_training(model_key, 'classification')
        return self

    ##
    ## Training Random Forest
    ##
    def train_random_forest(self, task='both', hyperparameter_tuning=False, show_viz=False):
        model_key = self._get_model_key('Random Forest', hyperparameter_tuning)
        
        if task in ['regression', 'both']:
            print(f"Training {model_key} Regressor...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                param_grid = {k: v for k, v in self.rf_params.items() if k != 'class_weight'}
                grid = GridSearchCV(RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                                  param_grid, cv=self.cv_folds, scoring='r2', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_reg, self.y_train_reg)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
                best_model.fit(self.X_train_reg, self.y_train_reg)
                best_params = {}

            y_pred = best_model.predict(self.X_test_reg)
            self.models['regression'][model_key] = best_model
            self.metrics['regression'][model_key] = {
                'r2': r2_score(self.y_test_reg, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'mae': mean_absolute_error(self.y_test_reg, y_pred),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }

            if show_viz:
                self._visualize_after_training(model_key, 'regression')

        if task in ['classification', 'both']:
            print(f"Training {model_key} Classifier...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                grid = GridSearchCV(RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                                  self.rf_params, cv=self.cv_folds, scoring='accuracy', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_clf, self.y_train_clf)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                best_model.fit(self.X_train_clf, self.y_train_clf)
                best_params = {}

            y_pred = best_model.predict(self.X_test_clf)
            self.models['classification'][model_key] = best_model
            self.metrics['classification'][model_key] = {
                'accuracy': accuracy_score(self.y_test_clf, y_pred),
                'f1': f1_score(self.y_test_clf, y_pred, average='weighted'),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }
            if show_viz:
                self._visualize_after_training(model_key, 'classification')            
        return self

    ##
    ## Training Support Vector Machine
    ##
    def train_svm(self, task='both', hyperparameter_tuning=False, show_viz=False):
        model_key = self._get_model_key('SVM', hyperparameter_tuning)
        
        if task in ['regression', 'both']:
            print(f"Training {model_key} Regressor...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                param_grid = {k: v for k, v in self.svm_params.items() if k != 'class_weight'}
                grid = GridSearchCV(SVR(), param_grid, cv=self.cv_folds, scoring='r2', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_reg_scaled, self.y_train_reg)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = SVR()
                best_model.fit(self.X_train_reg_scaled, self.y_train_reg)
                best_params = {}

            y_pred = best_model.predict(self.X_test_reg_scaled)
            self.models['regression'][model_key] = best_model
            self.metrics['regression'][model_key] = {
                'r2': r2_score(self.y_test_reg, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'mae': mean_absolute_error(self.y_test_reg, y_pred),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }
            if show_viz:
                self._visualize_after_training(model_key, 'regression')

        if task in ['classification', 'both']:
            print(f"Training {model_key} Classifier...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                grid = GridSearchCV(SVC(random_state=self.random_state), self.svm_params, cv=self.cv_folds,
                                  scoring='accuracy', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_clf_scaled, self.y_train_clf)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = SVC(random_state=self.random_state)
                best_model.fit(self.X_train_clf_scaled, self.y_train_clf)
                best_params = {}

            y_pred = best_model.predict(self.X_test_clf_scaled)
            self.models['classification'][model_key] = best_model
            self.metrics['classification'][model_key] = {
                'accuracy': accuracy_score(self.y_test_clf, y_pred),
                'f1': f1_score(self.y_test_clf, y_pred, average='weighted'),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }

            if show_viz:
                self._visualize_after_training(model_key, 'classification')
        return self

    ##
    ## Training Extreme Gradient Boost
    ##
    def train_xgboost(self, task='both', hyperparameter_tuning=False, show_viz=False):
        model_key = self._get_model_key('XGBoost', hyperparameter_tuning)
        
        if task in ['regression', 'both']:
            print(f"Training {model_key} Regressor...")
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                grid = GridSearchCV(XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs), self.xgb_params,
                                  cv=self.cv_folds, scoring='r2', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_reg, self.y_train_reg)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
                best_model.fit(self.X_train_reg, self.y_train_reg)
                best_params = {}

            y_pred = best_model.predict(self.X_test_reg)
            self.models['regression'][model_key] = best_model
            self.metrics['regression'][model_key] = {
                'r2': r2_score(self.y_test_reg, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'mae': mean_absolute_error(self.y_test_reg, y_pred),
                'best_params': best_params,
                'tuned': hyperparameter_tuning
            }
            if show_viz:
                self._visualize_after_training(model_key, 'regression')

        if task in ['classification', 'both']:
            print(f"Training {model_key} Classifier...")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(self.y_train_clf)
            if hyperparameter_tuning:
                print("---> Hyperparameter tuning...")
                grid = GridSearchCV(XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs), self.xgb_params,
                                  cv=self.cv_folds, scoring='accuracy', n_jobs=self.n_jobs, verbose=self.verbose)
                grid.fit(self.X_train_clf, y_train_encoded)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                best_model = XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                best_model.fit(self.X_train_clf, y_train_encoded)
                best_params = {}

            y_pred_encoded = best_model.predict(self.X_test_clf)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            self.models['classification'][model_key] = best_model
            self.metrics['classification'][model_key] = {
                'accuracy': accuracy_score(self.y_test_clf, y_pred),
                'f1': f1_score(self.y_test_clf, y_pred, average='weighted'),
                'best_params': best_params,
                'label_encoder': label_encoder,
                'tuned': hyperparameter_tuning
            }
            if show_viz:
                self._visualize_after_training(model_key, 'classification')
        return self

    ##
    ## Helper to train all needed models at once
    ##
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

    ##
    ## Helper to display confusion matrix
    ##
    def _confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    ##
    ## Helper for classification risk distribution
    ##
    def _classification_report(self, y_true, y_pred):
        y_true = np.array([str(y) for y in y_true])
        y_pred = np.array([str(y) for y in y_pred])
        report = classification_report(y_true, y_pred)
        print("Classification Report:")
        print(report)

    ##
    ## Helper for ROC curve
    ##
    def _roc_curve(self, y_true, y_pred_proba, model_name):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label = "ROC Curve")
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    ##
    ## Helper for the area under the ROC curve
    ##
    def _roc_auc_curve(self, y_true, y_pred_proba, model_name):
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC AUC - {model_name} - Score: {auc:.4f}")

    ##
    ## Helper for the visualization of the Precision Recall curve
    ##
    def _precision_recall_curve(self, y_true, y_pred_proba, model_name):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label="Precision-Recall Curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"{model_name} - Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    ##
    ## Helper for precision and recall information
    ##
    def _precision_recall(self, y_true, y_pred, model_name, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"):
        precision = precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight, zero_division=zero_division)
        print(f"Model: {model_name}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    ##
    ## Wrapper to display visualizations
    ##
    def _visualize_after_training(self, model_name, task):
        if task == 'regression':
            self.display_visualizations(model_name=model_name, task='regression')
        elif task == 'classification':
            self.display_visualizations(model_name=model_name, task='classification')

    ##
    ## Helper for visualizations
    ##
    def display_visualizations(self, model_name=None, task=None):
        if task == None or task == 'classification':
            # Use best classification model if model_name is not specified
            if model_name is None:
                if self.best_clf_name is None:
                    print("No classification models have been trained yet.")
                    return self
                model_name = self.best_clf_name
            
            if model_name not in self.models['classification']:
                print(f"Model '{model_name}' not found in trained classification models.")
                print(f"Available models: {list(self.models['classification'].keys())}")
                return self
            
            model = self.models['classification'][model_name]
            
            print("="*80)
            print(f"CLASSIFICATION VISUALIZATIONS FOR: {model_name}")
            print("="*80)
            
            # Especial case for SVM and XGBoost
            if 'SVM' in model_name:
                y_pred = model.predict(self.X_test_clf_scaled)
            elif 'XGBoost' in model_name:
                # XGBoost model predicts encoded labels, need to decode them
                y_pred_encoded = model.predict(self.X_test_clf)
                if 'label_encoder' in self.metrics['classification'][model_name]:
                    label_encoder = self.metrics['classification'][model_name]['label_encoder']
                    y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
                else:
                    y_pred = y_pred_encoded
            else:
                y_pred = model.predict(self.X_test_clf)
            
            # For binary classification metrics, we need to convert to binary
            # Check if this is a binary classification problem
            unique_classes = np.unique(self.y_test_clf)
            is_binary = len(unique_classes) == 2
            
            if is_binary:
                # Get probability predictions for ROC and PR curves
                if hasattr(model, 'predict_proba'):
                    if 'SVM' in model_name:
                        y_pred_proba = model.predict_proba(self.X_test_clf_scaled)[:, 1]
                    else:
                        y_pred_proba = model.predict_proba(self.X_test_clf)[:, 1]
                elif hasattr(model, 'decision_function'):
                    if 'SVM' in model_name:
                        y_pred_proba = model.decision_function(self.X_test_clf_scaled)
                    else:
                        y_pred_proba = model.decision_function(self.X_test_clf)
                else:
                    y_pred_proba = None
                
                # Convert labels to binary for metrics
                le = LabelEncoder()
                y_test_binary = le.fit_transform(self.y_test_clf)
                y_pred_binary = le.transform(y_pred)
            
            print("\n1. Confusion Matrix:")
            self._confusion_matrix(self.y_test_clf, y_pred, model_name=model_name)
            
            print("\n2. Precision and Recall Metrics:")
            self._precision_recall(self.y_test_clf, y_pred, average='weighted', model_name=model_name)
            
            if is_binary and y_pred_proba is not None:
                print("\n4. ROC Curve:")
                self._roc_curve(y_test_binary, y_pred_proba, model_name=model_name)
                
                print("\n5. ROC AUC Score:")
                self._roc_auc_curve(y_test_binary, y_pred_proba, model_name=model_name)
                
                print("\n6. Precision-Recall Curve:")
                self._precision_recall_curve(y_test_binary, y_pred_proba, model_name=model_name)
            
            print("\n" + "="*80)
            
        elif task == None or task == 'regression':
            # Use best regression model if model_name is not specified
            if model_name is None:
                if self.best_reg_name is None:
                    print("No regression models have been trained yet.")
                    return self
                model_name = self.best_reg_name

            if model_name not in self.models['regression']:
                print(f"Model '{model_name}' not found in trained regression models.")
                print(f"Available models: {list(self.models['regression'].keys())}")
                return self

            model = self.models['regression'][model_name]

            print("="*80)
            print(f"REGRESSION VISUALIZATIONS FOR: {model_name}")
            print("="*80)

            # Get predictions - handle SVM
            if 'SVM' in model_name:
                y_pred = model.predict(self.X_test_reg_scaled)
            else:
                y_pred = model.predict(self.X_test_reg)

            # Calculate residuals
            residuals = self.y_test_reg - y_pred

            # Get metrics
            r2 = self.metrics['regression'][model_name]['r2']
            rmse = self.metrics['regression'][model_name]['rmse']
            mae = self.metrics['regression'][model_name]['mae']

            print("\n1. Model Performance Metrics:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")

            print("\n2. Actual vs Predicted Values:")
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test_reg, y_pred, alpha=0.6, edgecolors='black', s=50)
            plt.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                    [self.y_test_reg.min(), self.y_test_reg.max()],
                    'r--', lw=2, label='Perfect Prediction')
            plt.xlabel('Actual ESG Score', fontsize=12)
            plt.ylabel('Predicted ESG Score', fontsize=12)
            plt.title(f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            print("\n3. Residuals Plot:")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', s=50)
            plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
            plt.xlabel('Predicted ESG Score', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.title(f'Residuals Plot - {model_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            print("\n4. Residuals Distribution:")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))


            axes[0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
            axes[0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
            axes[0].set_xlabel('Residuals', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Residuals Histogram', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            print("\n5. Error Metrics Summary:")
            mse = mean_squared_error(self.y_test_reg, y_pred)
            print(f"   Mean Squared Error (MSE): {mse:.4f}")
            print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"   Mean Absolute Error (MAE): {mae:.4f}")
            print(f"   R² Score: {r2:.4f}")

            # Calculate additional metrics
            mean_actual = np.mean(self.y_test_reg)
            print(f"   Mean Actual Value: {mean_actual:.4f}")
            print(f"   Mean Predicted Value: {np.mean(y_pred):.4f}")
            print(f"   Std of Residuals: {np.std(residuals):.4f}")

            print("\n6. Prediction Error Distribution:")
            plt.figure(figsize=(10, 6))
            errors = np.abs(residuals)
            plt.scatter(range(len(errors)), errors, alpha=0.6, edgecolors='black', s=50)
            plt.axhline(y=mae, color='r', linestyle='--', lw=2, label=f'MAE = {mae:.2f}')
            plt.axhline(y=rmse, color='orange', linestyle='--', lw=2, label=f'RMSE = {rmse:.2f}')
            plt.xlabel('Sample Index', fontsize=12)
            plt.ylabel('Absolute Error', fontsize=12)
            plt.title(f'Prediction Errors - {model_name}', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            print("\n" + "="*80)

        return self

    ##
    ## Helper to get the best models. Works for tuned and untuned (separately)
    ##
    def best_models(self):
        results = {'tuned': {}, 'untuned': {}}
        
        if self.metrics['regression']:
            # Separate tuned and untuned models
            tuned_reg = {k: v for k, v in self.metrics['regression'].items() if v.get('tuned', False)}
            untuned_reg = {k: v for k, v in self.metrics['regression'].items() if not v.get('tuned', False)}
            
            if tuned_reg:
                best_tuned_reg = max(tuned_reg.items(), key=lambda x: x[1]['r2'])
                self.best_reg_name = best_tuned_reg[0]
                results['tuned']['regression'] = {
                    'name': best_tuned_reg[0],
                    'model': self.models['regression'][best_tuned_reg[0]],
                    'metrics': best_tuned_reg[1]
                }
            
            if untuned_reg:
                best_untuned_reg = max(untuned_reg.items(), key=lambda x: x[1]['r2'])
                self.best_reg_name_untuned = best_untuned_reg[0]
                results['untuned']['regression'] = {
                    'name': best_untuned_reg[0],
                    'model': self.models['regression'][best_untuned_reg[0]],
                    'metrics': best_untuned_reg[1]
                }
        
        if self.metrics['classification']:
            # Separate tuned and untuned models
            tuned_clf = {k: v for k, v in self.metrics['classification'].items() if v.get('tuned', False)}
            untuned_clf = {k: v for k, v in self.metrics['classification'].items() if not v.get('tuned', False)}
            
            if tuned_clf:
                best_tuned_clf = max(tuned_clf.items(), key=lambda x: x[1]['f1'])
                self.best_clf_name = best_tuned_clf[0]
                results['tuned']['classification'] = {
                    'name': best_tuned_clf[0],
                    'model': self.models['classification'][best_tuned_clf[0]],
                    'metrics': best_tuned_clf[1]
                }
            
            if untuned_clf:
                best_untuned_clf = max(untuned_clf.items(), key=lambda x: x[1]['f1'])
                self.best_clf_name_untuned = best_untuned_clf[0]
                results['untuned']['classification'] = {
                    'name': best_untuned_clf[0],
                    'model': self.models['classification'][best_untuned_clf[0]],
                    'metrics': best_untuned_clf[1]
                }
        
        return results
    
    ##
    ## Helper for model comparison
    ##
    def print_model_comparison(self):
        print("="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        if self.metrics['regression']:
            # Separate untuned and tuned models
            untuned_reg = {k: v for k, v in self.metrics['regression'].items() if not v.get('tuned', False)}
            tuned_reg = {k: v for k, v in self.metrics['regression'].items() if v.get('tuned', False)}
            
            if untuned_reg:
                print("\nREGRESSION MODELS (UNTUNED):")
                print("-"*80)
                for name, metrics in sorted(untuned_reg.items(), key=lambda x: x[1]['r2'], reverse=True):
                    print(f"{name:30s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
            
            if tuned_reg:
                print("\nREGRESSION MODELS (TUNED):")
                print("-"*80)
                for name, metrics in sorted(tuned_reg.items(), key=lambda x: x[1]['r2'], reverse=True):
                    print(f"{name:30s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
                    if metrics['best_params']:
                        print(f"  Best Hyperparameters: {metrics['best_params']}")

        if self.metrics['classification']:
            # Separate untuned and tuned models
            untuned_clf = {k: v for k, v in self.metrics['classification'].items() if not v.get('tuned', False)}
            tuned_clf = {k: v for k, v in self.metrics['classification'].items() if v.get('tuned', False)}
            
            if untuned_clf:
                print("\nCLASSIFICATION MODELS (UNTUNED):")
                print("-"*80)
                for name, metrics in sorted(untuned_clf.items(), key=lambda x: x[1]['f1'], reverse=True):
                    print(f"{name:30s}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1']:.4f}")
            
            if tuned_clf:
                print("\nCLASSIFICATION MODELS (TUNED):")
                print("-"*80)
                for name, metrics in sorted(tuned_clf.items(), key=lambda x: x[1]['f1'], reverse=True):
                    print(f"{name:30s}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1']:.4f}")
                    if metrics.get('best_params'):
                        print(f"  Best Hyperparameters: {metrics['best_params']}")
        
        best = self.best_models()
        print("\n" + "="*80)
        print("BEST MODELS SUMMARY")
        print("="*80)
        
        if 'regression' in best['untuned']:
            print(f"\nBest Untuned Regression Model: {best['untuned']['regression']['name']}")
            print(f"  R² Score: {best['untuned']['regression']['metrics']['r2']:.4f}")
            print(f"  RMSE: {best['untuned']['regression']['metrics']['rmse']:.4f}")
            print(f"  MAE: {best['untuned']['regression']['metrics']['mae']:.4f}")
        
        if 'regression' in best['tuned']:
            print(f"\nBest Tuned Regression Model: {best['tuned']['regression']['name']}")
            print(f"  R² Score: {best['tuned']['regression']['metrics']['r2']:.4f}")
            print(f"  RMSE: {best['tuned']['regression']['metrics']['rmse']:.4f}")
            print(f"  MAE: {best['tuned']['regression']['metrics']['mae']:.4f}")
            if best['tuned']['regression']['metrics']['best_params']:
                print(f"  Best Hyperparameters: {best['tuned']['regression']['metrics']['best_params']}")
        
        if 'classification' in best['untuned']:
            print(f"\nBest Untuned Classification Model: {best['untuned']['classification']['name']}")
            print(f"  Accuracy: {best['untuned']['classification']['metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {best['untuned']['classification']['metrics']['f1']:.4f}")
        
        if 'classification' in best['tuned']:
            print(f"\nBest Tuned Classification Model: {best['tuned']['classification']['name']}")
            print(f"  Accuracy: {best['tuned']['classification']['metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {best['tuned']['classification']['metrics']['f1']:.4f}")
            if best['tuned']['classification']['metrics'].get('best_params'):
                print(f"  Best Hyperparameters: {best['tuned']['classification']['metrics']['best_params']}")
        
        return self
    
    ##
    ## Wrapper feature importance
    ##
    def feature_importance(self, task='both', top_n=15):
        results = {}
        if task in ['regression', 'both'] and self.best_reg_name:
            model = self.models['regression'][self.best_reg_name]
            if hasattr(model, 'feature_importances_'):
                results['regression'] = pd.DataFrame({
                    'Feature': self.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(top_n)
        if task in ['classification', 'both'] and self.best_clf_name:
            model = self.models['classification'][self.best_clf_name]
            if hasattr(model, 'feature_importances_'):
                results['classification'] = pd.DataFrame({
                    'Feature': self.feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(top_n)
        return results
    
    ##
    ## Helper for saving plots (especifically to use on my presentation :)
    ##
    def save_best_worst_plots(self, output_dir='plots', task='both'):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAVING BEST AND WORST MODEL PLOTS")
        print("="*80)
        
        # REGRESSION TASKS
        if task in ['regression', 'both'] and self.metrics['regression']:
            print("\nProcessing REGRESSION models...")
            
            # Get best and worst regression models based on R² score
            reg_models_sorted = sorted(self.metrics['regression'].items(),
                                      key=lambda x: x[1]['r2'],
                                      reverse=True)
            
            best_reg_name = reg_models_sorted[0][0]
            worst_reg_name = reg_models_sorted[-1][0]
            
            print(f"  Best model: {best_reg_name} (R²={reg_models_sorted[0][1]['r2']:.4f})")
            print(f"  Worst model: {worst_reg_name} (R²={reg_models_sorted[-1][1]['r2']:.4f})")
            
            for model_name in [best_reg_name, worst_reg_name]:
                model_type = "best" if model_name == best_reg_name else "worst"
                model = self.models['regression'][model_name]
                
                # Get predictions
                if 'SVM' in model_name:
                    y_pred = model.predict(self.X_test_reg_scaled)
                else:
                    y_pred = model.predict(self.X_test_reg)
                
                residuals = self.y_test_reg - y_pred
                
                # Get metrics
                r2 = self.metrics['regression'][model_name]['r2']
                rmse = self.metrics['regression'][model_name]['rmse']
                mae = self.metrics['regression'][model_name]['mae']
                
                # Sanitize filename
                safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                # 1. Actual vs Predicted Plot
                plt.figure(figsize=(10, 6))
                plt.scatter(self.y_test_reg, y_pred, alpha=0.6, edgecolors='black', s=50)
                plt.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                        [self.y_test_reg.min(), self.y_test_reg.max()],
                        'r--', lw=2, label='Perfect Prediction')
                plt.xlabel('Actual ESG Score', fontsize=12)
                plt.ylabel('Predicted ESG Score', fontsize=12)
                plt.title(f'Actual vs Predicted - {model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}',
                         fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/reg_{model_type}_{safe_model_name}_actual_vs_predicted.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Residuals Plot
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', s=50)
                plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
                plt.xlabel('Predicted ESG Score', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title(f'Residuals Plot - {model_name}\nR²={r2:.4f}',
                         fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/reg_{model_type}_{safe_model_name}_residuals.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved plots for {model_type} model: {model_name}")
        
        # CLASSIFICATION TASKS
        if task in ['classification', 'both'] and self.metrics['classification']:
            print("\nProcessing CLASSIFICATION models...")
            
            # Get best and worst classification models based on F1 score
            clf_models_sorted = sorted(self.metrics['classification'].items(),
                                      key=lambda x: x[1]['f1'],
                                      reverse=True)
            
            best_clf_name = clf_models_sorted[0][0]
            worst_clf_name = clf_models_sorted[-1][0]
            
            print(f"  Best model: {best_clf_name} (F1={clf_models_sorted[0][1]['f1']:.4f})")
            print(f"  Worst model: {worst_clf_name} (F1={clf_models_sorted[-1][1]['f1']:.4f})")
            
            for model_name in [best_clf_name, worst_clf_name]:
                model_type = "best" if model_name == best_clf_name else "worst"
                model = self.models['classification'][model_name]
                
                # Get predictions
                if 'SVM' in model_name:
                    y_pred = model.predict(self.X_test_clf_scaled)
                elif 'XGBoost' in model_name:
                    y_pred_encoded = model.predict(self.X_test_clf)
                    if 'label_encoder' in self.metrics['classification'][model_name]:
                        label_encoder = self.metrics['classification'][model_name]['label_encoder']
                        y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
                    else:
                        y_pred = y_pred_encoded
                else:
                    y_pred = model.predict(self.X_test_clf)
                
                # Sanitize filename
                safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                # 1. Confusion Matrix
                cm = confusion_matrix(self.y_test_clf, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
                plt.savefig(f'{output_dir}/class_{model_type}_{safe_model_name}_confusion_matrix.png',
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Check if binary classification for ROC/PR curves
                unique_classes = np.unique(self.y_test_clf)
                is_binary = len(unique_classes) == 2
                
                if is_binary:
                    # Get probability predictions
                    if hasattr(model, 'predict_proba'):
                        if 'SVM' in model_name:
                            y_pred_proba = model.predict_proba(self.X_test_clf_scaled)[:, 1]
                        else:
                            y_pred_proba = model.predict_proba(self.X_test_clf)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        if 'SVM' in model_name:
                            y_pred_proba = model.decision_function(self.X_test_clf_scaled)
                        else:
                            y_pred_proba = model.decision_function(self.X_test_clf)
                    else:
                        y_pred_proba = None
                    
                    if y_pred_proba is not None:
                        # Convert to binary
                        le = LabelEncoder()
                        y_test_binary = le.fit_transform(self.y_test_clf)
                        
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                        auc = roc_auc_score(y_test_binary, y_pred_proba)
                        
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC={auc:.4f})")
                        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight='bold')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f'{output_dir}/class_{model_type}_{safe_model_name}_roc_curve.png',
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                        auc = roc_auc_score(y_test_binary, y_pred_proba)
                        
                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC={auc:.4f})")
                        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f"{model_name} - ROC Curve", fontsize=14, fontweight='bold')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f'{output_dir}/class_{model_type}_{safe_model_name}_roc_curve.png',
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Saved 3 plots for {model_type} model: {model_name}")
                    else:
                        print(f"  Saved 1 plot for {model_type} model: {model_name} (no probability predictions)")
                else:
                    print(f"  Saved 1 plot for {model_type} model: {model_name} (multiclass - no ROC/PR curves)")
        
        print("\n" + "="*80)
        print(f"All plots saved to directory: {output_dir}/")
        print("="*80 + "\n")
        
        return self
    
    ##
    ## Helper to display the plots for the best and worse ones (especifically to use on my presentation :)
    ##
    def display_best_worst_plots(self, task='both'):
        print("\n" + "="*80)
        print("DISPLAYING BEST AND WORST MODEL PLOTS (UNTUNED AND TUNED)")
        print("="*80)

        # REGRESSION TASK
        if task in ['regression', 'both'] and self.metrics['regression']:
            print("\nProcessing REGRESSION models...")

            # Separate tuned and untuned regression models
            tuned_reg = {k: v for k, v in self.metrics['regression'].items() if v.get('tuned', False)}
            untuned_reg = {k: v for k, v in self.metrics['regression'].items() if not v.get('tuned', False)}

            # Get best and worst models based on R² score
            models_to_display = []

            # 1. Best Regression Untuned Model
            if untuned_reg:
                best_untuned_reg = max(untuned_reg.items(), key=lambda x: x[1]['r2'])
                models_to_display.append((best_untuned_reg[0], 'Best Regression Untuned Model', best_untuned_reg[1]))
                print(f"  Best Untuned model: {best_untuned_reg[0]} (R²={best_untuned_reg[1]['r2']:.4f})")

            # 2. Worst Regression Untuned Model
            if untuned_reg:
                worst_untuned_reg = min(untuned_reg.items(), key=lambda x: x[1]['r2'])
                models_to_display.append((worst_untuned_reg[0], 'Worst Regression Untuned Model', worst_untuned_reg[1]))
                print(f"  Worst Untuned model: {worst_untuned_reg[0]} (R²={worst_untuned_reg[1]['r2']:.4f})")

            # 3. Best Regression Tuned Model
            if tuned_reg:
                best_tuned_reg = max(tuned_reg.items(), key=lambda x: x[1]['r2'])
                models_to_display.append((best_tuned_reg[0], 'Best Regression Tuned Model', best_tuned_reg[1]))
                print(f"  Best Tuned model: {best_tuned_reg[0]} (R²={best_tuned_reg[1]['r2']:.4f})")

            # 4. Worst Regression Tuned Model
            if tuned_reg:
                worst_tuned_reg = min(tuned_reg.items(), key=lambda x: x[1]['r2'])
                models_to_display.append((worst_tuned_reg[0], 'Worst Regression Tuned Model', worst_tuned_reg[1]))
                print(f"  Worst Tuned model: {worst_tuned_reg[0]} (R²={worst_tuned_reg[1]['r2']:.4f})")

            for model_name, model_type, metrics in models_to_display:
                model = self.models['regression'][model_name]

                # Get predictions
                if 'SVM' in model_name:
                    y_pred = model.predict(self.X_test_reg_scaled)
                else:
                    y_pred = model.predict(self.X_test_reg)

                residuals = self.y_test_reg - y_pred

                # Get metrics
                r2 = metrics['r2']
                rmse = metrics['rmse']
                mae = metrics['mae']

                print(f"\n  Displaying plots for {model_type}: {model_name}")

                # Create a 2x2 grid of plots
                fig = plt.figure(figsize=(18, 14))

                # 1. Actual vs Predicted Plot
                ax1 = plt.subplot(2, 2, 1)
                ax1.scatter(self.y_test_reg, y_pred, alpha=0.6, edgecolors='black', s=50)
                ax1.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                        [self.y_test_reg.min(), self.y_test_reg.max()],
                        'r--', lw=2, label='Perfect Prediction')
                ax1.set_xlabel('Actual ESG Score', fontsize=12)
                ax1.set_ylabel('Predicted ESG Score', fontsize=12)
                ax1.set_title(f'Actual vs Predicted\nR²={r2:.4f}, RMSE={rmse:.4f}',
                         fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 2. Residuals Plot
                ax2 = plt.subplot(2, 2, 2)
                ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', s=50)
                ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residual')
                ax2.set_xlabel('Predicted ESG Score', fontsize=12)
                ax2.set_ylabel('Residuals', fontsize=12)
                ax2.set_title(f'Residuals Plot\nR²={r2:.4f}',
                         fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 3. Residuals Histogram
                ax3 = plt.subplot(2, 2, 3)
                ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
                ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
                ax3.set_xlabel('Residuals', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.set_title('Residuals Histogram', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # 4. Q-Q Plot
                ax4 = plt.subplot(2, 2, 4)
                stats.probplot(residuals, dist="norm", plot=ax4)
                ax4.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)

                plt.suptitle(f'{model_type}: {model_name}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}',
                            fontsize=16, fontweight='bold', y=0.995)
                plt.tight_layout()
                plt.show()

        # CLASSIFICATION TASK
        if task in ['classification', 'both'] and self.metrics['classification']:
            print("\nProcessing CLASSIFICATION models...")

            # Separate tuned and untuned classification models
            tuned_clf = {k: v for k, v in self.metrics['classification'].items() if v.get('tuned', False)}
            untuned_clf = {k: v for k, v in self.metrics['classification'].items() if not v.get('tuned', False)}

            # Get best and worst models based on F1 score
            models_to_display = []

            # 5. Best Classification Untuned Model
            if untuned_clf:
                best_untuned_clf = max(untuned_clf.items(), key=lambda x: x[1]['f1'])
                models_to_display.append((best_untuned_clf[0], 'Best Classification Untuned Model', best_untuned_clf[1]))
                print(f"  Best Untuned model: {best_untuned_clf[0]} (F1={best_untuned_clf[1]['f1']:.4f})")

            # 6. Worst Classification Untuned Model
            if untuned_clf:
                worst_untuned_clf = min(untuned_clf.items(), key=lambda x: x[1]['f1'])
                models_to_display.append((worst_untuned_clf[0], 'Worst Classification Untuned Model', worst_untuned_clf[1]))
                print(f"  Worst Untuned model: {worst_untuned_clf[0]} (F1={worst_untuned_clf[1]['f1']:.4f})")

            # 7. Best Classification Tuned Model
            if tuned_clf:
                best_tuned_clf = max(tuned_clf.items(), key=lambda x: x[1]['f1'])
                models_to_display.append((best_tuned_clf[0], 'Best Classification Tuned Model', best_tuned_clf[1]))
                print(f"  Best Tuned model: {best_tuned_clf[0]} (F1={best_tuned_clf[1]['f1']:.4f})")

            # 8. Worst Classification Tuned Model
            if tuned_clf:
                worst_tuned_clf = min(tuned_clf.items(), key=lambda x: x[1]['f1'])
                models_to_display.append((worst_tuned_clf[0], 'Worst Classification Tuned Model', worst_tuned_clf[1]))
                print(f"  Worst Tuned model: {worst_tuned_clf[0]} (F1={worst_tuned_clf[1]['f1']:.4f})")

            for model_name, model_type, metrics in models_to_display:
                model = self.models['classification'][model_name]

                # Get predictions
                if 'SVM' in model_name:
                    y_pred = model.predict(self.X_test_clf_scaled)
                elif 'XGBoost' in model_name:
                    y_pred_encoded = model.predict(self.X_test_clf)
                    if 'label_encoder' in self.metrics['classification'][model_name]:
                        label_encoder = self.metrics['classification'][model_name]['label_encoder']
                        y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
                    else:
                        y_pred = y_pred_encoded
                else:
                    y_pred = model.predict(self.X_test_clf)

                print(f"\n  Displaying plots for {model_type}: {model_name}")

                # Get metrics
                accuracy = metrics['accuracy']
                f1 = metrics['f1']

                # Check if binary classification for PR curves
                unique_classes = np.unique(self.y_test_clf)
                is_binary = len(unique_classes) == 2

                if is_binary:
                    # Get probability predictions
                    if hasattr(model, 'predict_proba'):
                        if 'SVM' in model_name:
                            y_pred_proba = model.predict_proba(self.X_test_clf_scaled)[:, 1]
                        else:
                            y_pred_proba = model.predict_proba(self.X_test_clf)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        if 'SVM' in model_name:
                            y_pred_proba = model.decision_function(self.X_test_clf_scaled)
                        else:
                            y_pred_proba = model.decision_function(self.X_test_clf)
                    else:
                        y_pred_proba = None

                    if y_pred_proba is not None:
                        le = LabelEncoder()
                        y_test_binary = le.fit_transform(self.y_test_clf)

                        # Create 1x2 grid for binary classification (Confusion Matrix and PR Curve only, NO ROC/AUC)
                        fig = plt.figure(figsize=(14, 6))

                        # 1. Confusion Matrix
                        ax1 = plt.subplot(1, 2, 1)
                        cm = confusion_matrix(self.y_test_clf, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
                        ax1.set_xlabel('Predicted')
                        ax1.set_ylabel('True')
                        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

                        # 2. Precision-Recall Curve (NO ROC/AUC plot)
                        ax2 = plt.subplot(1, 2, 2)
                        precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)
                        ax2.plot(recall, precision, color='blue', label="PR Curve")
                        ax2.set_xlabel('Recall')
                        ax2.set_ylabel('Precision')
                        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)

                        plt.suptitle(f'{model_type}: {model_name}\nAccuracy={accuracy:.4f}, F1-Score={f1:.4f}',
                                    fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        plt.show()
                    else:
                        # Just confusion matrix
                        fig = plt.figure(figsize=(8, 6))
                        cm = confusion_matrix(self.y_test_clf, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'{model_type}: {model_name}\nAccuracy={accuracy:.4f}, F1-Score={f1:.4f}',
                                 fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.show()
                else:
                    # Multiclass - just confusion matrix
                    fig = plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(self.y_test_clf, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'{model_type}: {model_name}\nAccuracy={accuracy:.4f}, F1-Score={f1:.4f}',
                             fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.show()

        print("\n" + "="*80)
        print("All plots displayed!")
        print("="*80 + "\n")

        return self

    ##
    ## Helper to display a summary of the project
    ##
    def summary(self, display_plots=False, save_plots=False, plots_dir='plots'):
        print("="*80)
        print("ESG MODEL TRAINER - COMPREHENSIVE SUMMARY")
        print("="*80)
        print("\nDATA INFORMATION:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Training samples (regression): {len(self.X_train_reg)}")
        print(f"  Test samples (regression): {len(self.X_test_reg)}")
        print(f"  Training samples (classification): {len(self.X_train_clf)}")
        print(f"  Test samples (classification): {len(self.X_test_clf)}")
        print("\nRISK DISTRIBUTION:")
        print(self.df['esg_risk'].value_counts().sort_index())
        print("\nHYPERPARAMETERS:")
        print(f"  CV Folds: {self.cv_folds}")
        print(f"  Random State: {self.random_state}")
        print(f"  Percentiles: {self.percentiles}")
        self.print_model_comparison()
        
        # Display plots if requested
        if display_plots:
            self.display_best_worst_plots(task='both')
        
        # Save plots if requested
        if save_plots:
            self.save_best_worst_plots(output_dir=plots_dir, task='both')
        
        return self
    
    ##
    ## Helper to plot predictions
    ##
    def plot_predictions(self, task='regression'):
        if task == 'regression':
            # Get all available regression models
            available_models = list(self.models['regression'].keys())
            
            if not available_models:
                print("No regression models trained yet.")
                return self
            
            n_models = len(available_models)
            n_cols = 2
            n_rows = (n_models + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, model_name in enumerate(available_models):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]
                
                model = self.models['regression'][model_name]
                if 'SVM' in model_name:
                    y_pred = model.predict(self.X_test_reg_scaled)
                else:
                    y_pred = model.predict(self.X_test_reg)
                
                ax.scatter(self.y_test_reg, y_pred, alpha=0.6, edgecolors='black', s=50)
                ax.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                       [self.y_test_reg.min(), self.y_test_reg.max()], 'r--', lw=2, label='Perfect Prediction')
                
                r2 = self.metrics['regression'][model_name]['r2']
                rmse = self.metrics['regression'][model_name]['rmse']
                
                ax.set_xlabel('Actual ESG Score', fontsize=11)
                ax.set_ylabel('Predicted ESG Score', fontsize=11)
                ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontsize=10, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            for idx in range(n_models, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            plt.show()
        
        elif task == 'classification':
            available_models = list(self.models['classification'].keys())
            
            if not available_models:
                print("No classification models trained yet.")
                return self
            
            print("Classification Model Performance:")
            print("="*80)
            for model_name in available_models:
                metrics = self.metrics['classification'][model_name]
                tuning_status = "TUNED" if metrics.get('tuned', False) else "UNTUNED"
                print(f"{model_name:30s} [{tuning_status}]: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1']:.4f}")
        
        return self