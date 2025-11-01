# Predicting ESG Score and Sustainability Risk

Master's degree coursework project for **Introduction to Supervised Learning** at the University of Colorado Boulder.

## Overview

This project showcases supervised learning techniques and models to perform both **Regression** and **Classification** tasks on real-world ESG data. The implementation includes comprehensive data preprocessing, feature engineering, model training, hyperparameter optimization, and performance evaluation.

## Problem Statement

This project addresses two supervised learning tasks:

1. **Regression Analysis**: Predict the continuous `esg_score` variable, representing a company's Environmental, Social, and Governance performance. Higher scores indicate better ESG performance.
2. **Classification Task**: Classify companies into Sustainability Risk categories based on the `esg_risk` feature: LOW, MEDIUM-LOW, MEDIUM, MEDIUM-HIGH, and HIGH.

## Dataset

The dataset originates from the [**GHG Shopper**](https://www.ghgshopper.org) and [**Stakeholder Takeover**](https://www.stakeholdertakeover.org) initiatives, directed by [**Prof. Lynn M. LoPucki**](https://lopucki.com) of **UCLA Law School**. I, **Dyego Fernandes de Sousa**, contributed to these projects as a Fullstack Developer and received permission to use this proprietary dataset for academic purposes.

### Dataset Characteristics

- **Original format**: MS Access database converted to CSV
- **Data quality**: Raw dataset with non-normalized column names and extensive missing values
- **Content**: Greenhouse gas emissions data (Scope 1+2 Total), revenue information, industry classifications, brand counts, and reporting status
- **Size**: 251 companies in original dataset (synthetic data generation available via `generate_synthetic_data.py`)

### Target Variable Engineering

The dataset lacks pre-defined target variables, requiring custom feature engineering:

**ESG Score Calculation (`esg_score_raw`):**
- **Companies with reported emissions**: `score = scope1+2total / revenues`
- **Non-reporting companies** (excluding "realzero"): Penalized score using `(max_scope + 1) / (revenues × 0.75)`
- **RealZero companies**: Special handling based on brand count to avoid artificially perfect scores
- **Exclusions penalty**: Optional 10% penalty multiplier for companies with exclusions

**Normalized ESG Score (`esg_score` - Regression Target):**
- Applies **logarithmic transformation**: `log₁₊ₓ(esg_score_raw)` to handle extreme right-skewness
- **Min-max normalization** to scale scores to 0-100 range
- **Inverted scale**: Higher scores indicate better ESG performance

**Why Log Transformation?**
The raw emission-to-revenue ratios exhibit extreme right-skewness with outliers spanning 6+ orders of magnitude (0.009 to 3,412,000). Linear normalization compresses 99% of values near zero, making meaningful risk classification impossible. Log transformation compresses the range while preserving ordering, creating well-separated risk categories.

**Risk Classification (`esg_risk` - Classification Target):**
Percentile-based classification into 5 risk categories:
- **HIGH**: Non-reporting companies (non-realzero) or scores >78th percentile
- **MEDIUM-HIGH**: Scores between 55th-78th percentile
- **MEDIUM**: Scores between 23rd-55th percentile
- **MEDIUM-LOW**: Scores between 5th-23rd percentile
- **LOW**: Scores <5th percentile

## Project Structure

```
supervised_learning/
├── supervised_learning_final.ipynb    # Main analysis notebook
├── supervised_learning.py             # SupervisedLearning class implementation
├── generate_synthetic_data.py         # Synthetic data generator
├── ghg_data.csv                       # Original GHG emissions dataset
├── synthetic_ghg_data.csv             # Generated synthetic dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── presentation/                      # Presentation materials
    ├── esg_presentation_final.pdf     # Final presentation slides
    ├── esg_presentation_final.tex     # LaTeX source for presentation
    ├── compile_presentation.sh        # Unix/Linux compilation script
    ├── compile_presentation.bat       # Windows compilation script
    ├── code/                          # Code snippets for presentation
    │   ├── train_models.py
    │   ├── train_models_call.py
    │   ├── split_dataset.py
    │   └── bonus.py
    └── figures/                       # Generated visualization figures
        ├── histogram_before.png       # Pre-transformation distributions
        ├── histogram_after.png        # Post-transformation distributions
        ├── correlation_002.png        # Correlation heatmap
        ├── distribution_001.png       # ESG score distributions
        ├── reg_tuned_best.png         # Best tuned regression results
        ├── reg_tuned_worst.png        # Worst tuned regression results
        ├── class_tuned_best.png       # Best tuned classification results
        └── class_tuned_worst.png      # Worst tuned classification results
```

### Key Files

- **[supervised_learning_final.ipynb](supervised_learning_final.ipynb)**: Main analysis notebook with step-by-step implementation
- **[supervised_learning.py](supervised_learning.py)**: `SupervisedLearning` class containing all preprocessing, training, and evaluation methods
- **[ghg_data.csv](ghg_data.csv)**: Original proprietary GHG emissions dataset (251 companies)
- **[generate_synthetic_data.py](generate_synthetic_data.py)**: Utility script for generating synthetic training data (see Appendix)
- **[presentation/](presentation/)**: LaTeX presentation with compiled PDF and supporting figures

## Machine Learning Models

All models are trained both **with and without hyperparameter tuning** to demonstrate the impact of optimization:

### Tree-Based Models

**Decision Tree**
- Baseline model for both regression and classification
- Interpretable, rule-based predictions
- Hyperparameters: max_depth, min_samples_split, min_samples_leaf, max_features, class_weight

**Random Forest**
- Ensemble method using bootstrap aggregation (bagging)
- Reduces overfitting through averaging multiple trees
- Provides feature importance rankings
- Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap, max_features, class_weight

**AdaBoost (Adaptive Boosting)**
- Sequential ensemble method focusing on misclassified instances
- Adaptively weights weak learners
- Hyperparameters: n_estimators, learning_rate, loss (regression only)

**XGBoost (Extreme Gradient Boosting)**
- Advanced gradient boosting framework with built-in regularization
- *Note: Not taught in course, but included for comparison*
- Hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree

### Non-Tree Model

**Support Vector Machine (SVM)**
- Alternative non-tree, non-ensemble approach
- Effective in high-dimensional spaces with kernel-based transformations
- Uses scaled features (StandardScaler)
- Hyperparameters: C, gamma, kernel, class_weight

## Model Optimization

**Hyperparameter Tuning:**
- **GridSearchCV** with exhaustive parameter search over specified grids
- **3-fold cross-validation** for robust evaluation
- Separate parameter grids for regression and classification tasks
- Training performed both **with and without tuning** for comparison

**Evaluation Metrics:**
- **Regression**: R² Score, RMSE (Root Mean Squared Error), MAE (Mean Absolute Error)
- **Classification**: Accuracy, F1-Score (weighted average), Confusion Matrix, Precision-Recall curves

## Data Preprocessing Pipeline

The preprocessing pipeline transforms raw, inconsistent data into clean, normalized features suitable for machine learning:

1. **Data Loading with Encoding Handling**: Robust CSV parsing with automatic fallback through multiple encodings (UTF-8, Latin-1, CP1252)

2. **Column Normalization**: Standardizes column names to lowercase with underscores (e.g., "Scope1+2Total" → "scope1+2total")

3. **Target Variable Engineering**: Creates both `esg_score` (regression) and `esg_risk` (classification) targets from raw emission and revenue data

4. **Missing Value Treatment**:
   - Identifies and removes columns with 100% missing values
   - Applies median imputation (`SimpleImputer`) for partial missing data
   - Strategic handling distinguishes legitimate zeros from missing emissions

5. **Feature Scaling**:
   - **StandardScaler** applied separately for regression and classification to prevent data leakage
   - Fitted only on training data, then applied to test data
   - SVM models use scaled data; tree-based models use unscaled (scale-invariant)

6. **Logarithmic Transformation**: `log₁₊ₓ` transformation handles extreme right-skewness in emission ratios

7. **Train-Test Split**:
   - 80/20 split with `random_state=42` for reproducibility
   - **Stratified sampling** for classification to maintain class balance
   - Separate splits for regression and classification tasks

## Feature Selection

The feature selection strategy balances domain expertise with data-driven insights:

**Domain-Driven Exclusion:**
- Time-series financial data (2020-2021 sales, dates)
- Metadata and documentation fields (comments, URLs, logos)
- Administrative identifiers (ranking columns, parent company names)
- High-cardinality categorical features (company names, subsidiary info)

**Post-EDA Reduction:**
- Revenue/emission area codes with minimal correlation
- Disaggregated emission components (Scope 1 and Scope 2 individual values)
- Redundant emission fields

**Final Feature Set:**
After preprocessing, the models use 3 primary features:
- `scope1+2total`: Total greenhouse gas emissions (Scope 1 + Scope 2)
- `scope2emitundifferentiated`: Undifferentiated Scope 2 emissions
- `brands`: Number of brands owned by the company

This systematic approach reduces dimensionality from 50+ raw columns to 3 highly predictive features, improving model interpretability and reducing overfitting risk.

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone this repository and navigate to the project directory:
```bash
git clone <repository-url>
cd supervised_learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook

The primary analysis is conducted in the Jupyter notebook:

```bash
jupyter notebook supervised_learning_final.ipynb
```

**Notebook Structure:**
1. **Data Loading**: Loads `ghg_data.csv` with robust encoding handling
2. **Log Transformation**: Applies `log₁₊ₓ` transformation to handle extreme skewness
3. **Exploratory Data Analysis (EDA)**: Visualizations of ESG score distribution, emissions vs. score, correlation heatmap
4. **Feature Preparation**: Feature selection and 80/20 train-test split
5. **Model Training**: Trains all 5 models (DT, RF, AdaBoost, XGBoost, SVM) with and without hyperparameter tuning
6. **Model Evaluation**: Comprehensive visualizations, confusion matrices, residual plots, feature importance
7. **Summary**: Final performance comparison and best model selection

### Working with the SupervisedLearning Class

The `supervised_learning.py` file contains the `SupervisedLearning` class with all core functionality. Key methods include:

- `load_data()`: Load and normalize dataset columns
- `calculate_esg_score()`: Generate ESG scores from emissions and revenue
- `classify_risk()`: Create risk categories based on percentiles
- `prepare_features()` / `split_data()`: Feature selection and train-test splitting
- `train_models()`: Train specific models with/without tuning
- `print_model_comparison()`: Display performance metrics for all trained models
- `feature_importance()`: Extract and rank feature importance
- `display_visualizations()`: Generate comprehensive evaluation plots
- `summary()`: Complete project summary with results

## Dependencies

Core libraries used in this project:

- **pandas** (>= 2.0.0): Data manipulation and analysis
- **numpy** (>= 1.24.0): Numerical computing and array operations
- **scikit-learn** (>= 1.3.0): Machine learning algorithms, preprocessing, and evaluation
- **xgboost** (>= 2.0.0): Gradient boosting framework
- **matplotlib** (>= 3.7.0): Data visualization and plotting
- **seaborn** (>= 0.12.0): Statistical data visualization
- **scipy** (>= 1.10.0): Scientific computing (for Q-Q plots)
- **jupyter** (>= 1.0.0): Interactive notebook environment

See [requirements.txt](requirements.txt) for the complete dependency list.

## Results Overview

The project trains 5 different models (Decision Tree, Random Forest, AdaBoost, XGBoost, SVM) both with and without hyperparameter tuning, resulting in 10 regression models and 10 classification models.

**Performance varies based on dataset characteristics:**
- With synthetic data generation for larger datasets, ensemble methods (XGBoost, Random Forest, AdaBoost) typically achieve the highest performance
- With the original small dataset (251 samples), simpler models may perform better due to limited training data
- Hyperparameter tuning consistently improves model performance across all algorithms

**Key Findings:**
- Log transformation is critical for handling extreme emission-to-revenue ratio skewness
- Feature importance analysis reveals `scope1+2total` (total emissions) and `brands` as the most predictive features
- Tree-based ensemble methods effectively capture non-linear relationships in ESG data
- Classification achieves high accuracy (>95%) with tuned models on balanced synthetic data

## Academic Context

**Course:** Introduction to Supervised Learning
**Institution:** University of Colorado Boulder
**Program:** Master's Degree in Data Science
**Student:** Dyego Fernandes de Sousa

### Presentation

A comprehensive LaTeX presentation documenting the project methodology, results, and insights is available in the [presentation/](presentation/) directory. The presentation includes:

- **PDF version**: [esg_presentation_final.pdf](presentation/esg_presentation_final.pdf) - Ready-to-view compiled slides
- **LaTeX source**: [esg_presentation_final.tex](presentation/esg_presentation_final.tex) - Editable presentation source
- **Compilation scripts**: Automated scripts for compiling the presentation on Unix/Linux (`compile_presentation.sh`) or Windows (`compile_presentation.bat`)
- **Supporting materials**: Code snippets and visualization figures used in the presentation

## Acknowledgments

- **Prof. Lynn M. LoPucki** (UCLA Law School) - Director of [GHG Shopper](https://www.ghgshopper.org) and [Stakeholder Takeover](https://www.stakeholdertakeover.org) initiatives, and for granting permission to use the proprietary dataset for academic purposes
- University of Colorado Boulder - Course instruction and academic support
- [GHG Shopper](https://www.ghgshopper.org) and [Stakeholder Takeover](https://www.stakeholdertakeover.org) projects for providing real-world ESG data

## License

This project is intended for **academic purposes only**. The GHG emissions dataset is proprietary and used with explicit permission from Prof. Lynn M. LoPucki. Unauthorized commercial use is prohibited. Please contact the original data providers for any non-academic applications.

---

## Appendix: Synthetic Data Generation

For testing, development, or expanding the training dataset, the `generate_synthetic_data.py` script generates synthetic ESG data matching the original dataset's statistical distribution.

**Usage:**
```bash
# Generate 1000 synthetic samples (default)
python generate_synthetic_data.py

# Generate custom number of samples
python generate_synthetic_data.py 2500
```

**Output:**
- Creates `synthetic_ghg_data.csv` containing original data, copies, and synthetic records
- All synthetic company names are prefixed with `SYNTHETIC:` for identification
- Maintains realistic emission-to-revenue ratio distributions across 6 risk classes
- Preserves missing value patterns and feature correlations from the original dataset

**Note:** The synthetic data generator creates data with clean class separation optimized for demonstrating supervised learning techniques. Real-world ESG data contains significantly more complexity and noise.
