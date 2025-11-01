#!/usr/bin/env python3
"""
Synthetic ESG Data Generator

This script generates synthetic data based on the ghg_data.csv structure.
The final dataset includes:
- Original dataset rows (no prefix)
- 4 copies of the original dataset with "COPY:" prefix
- Synthetic data rows with "SYNTHETIC:" prefix

Requirements:
    pip install pandas numpy

Usage:
    python generate_synthetic_data.py [num_samples]

    num_samples: Number of synthetic rows to generate (default: 1000)

Examples:
    python generate_synthetic_data.py           # Generates 1000 rows (default)
    python generate_synthetic_data.py 500       # Generates 500 rows
    python generate_synthetic_data.py 2000      # Generates 2000 rows

Output:
    Creates 'synthetic_ghg_data.csv' containing:
    - Original data (no prefix)
    - 4 copies of original data (COPY: prefix)
    - Requested number of synthetic records (SYNTHETIC: prefix)
    Total rows = original + (4 * original) + num_samples
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def load_original_data(filepath='ghg_data.csv'):
    """Load the original GHG data with proper encoding handling."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='cp1252')
    return df

def generate_synthetic_data(original_df, num_samples=100, random_seed=42):
    """
    Generate synthetic data based on the original dataset statistics.

    Industry and revenue area influences:
    - Industry types influence both Scope1+2 Total emissions and Revenues with random multipliers
    - Revenue brackets (low/medium/high) apply additional random multipliers to both metrics
    - Emissions are also influenced by revenue levels (larger companies = higher emissions)
    - All multipliers are randomized to create realistic variation

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original GHG dataset
    num_samples : int
        Number of synthetic samples to generate
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with same structure as original
    """
    np.random.seed(random_seed)

    # Get numeric columns statistics
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns

    # Create synthetic dataframe
    synthetic_df = pd.DataFrame()

    # Generate ID column
    synthetic_df['ID'] = range(1, num_samples + 1)

    # Generate company names with SYNTHETIC: prefix
    company_types = ['Foods', 'Beverages', 'Dairy', 'Industries', 'Corporation',
                     'Inc.', 'Group', 'Holdings', 'Brands', 'Farms']
    company_names = []
    for i in range(num_samples):
        base_name = f"Company {i+1}"
        company_type = np.random.choice(company_types)
        company_names.append(f"SYNTHETIC: {base_name} {company_type}")

    synthetic_df['Company GHG Name'] = company_names

    # Industry types from original data
    industries = original_df['SasbIndustry'].dropna().unique()
    synthetic_df['SasbIndustry'] = np.random.choice(industries, size=num_samples)

    # Define industry-based multipliers for emissions and revenues
    # These will influence Scope1+2 Total and Revenues based on industry type
    industry_emission_multipliers = {
        'Food & Beverage Processing': np.random.uniform(0.8, 1.5, num_samples),
        'Agricultural Products': np.random.uniform(0.9, 1.6, num_samples),
        'Meat, Poultry & Dairy': np.random.uniform(1.2, 2.0, num_samples),
        'Processed Foods': np.random.uniform(0.7, 1.3, num_samples),
    }

    industry_revenue_multipliers = {
        'Food & Beverage Processing': np.random.uniform(0.8, 1.4, num_samples),
        'Agricultural Products': np.random.uniform(0.6, 1.2, num_samples),
        'Meat, Poultry & Dairy': np.random.uniform(0.9, 1.5, num_samples),
        'Processed Foods': np.random.uniform(0.7, 1.3, num_samples),
    }

    # Default multipliers for unknown industries
    default_emission_mult = np.random.uniform(0.8, 1.5, num_samples)
    default_revenue_mult = np.random.uniform(0.7, 1.3, num_samples)

    # Generate numeric columns EXCEPT Scope1+2Total (we'll do that last)
    for col in numeric_cols:
        if col == 'ID' or col == 'Scope1+2Total':
            continue

        # Get statistics from original data (excluding NaN)
        orig_data = original_df[col].dropna()

        if len(orig_data) == 0:
            # If column is all NaN in original, keep it NaN in synthetic
            synthetic_df[col] = np.nan
            continue

        # Calculate statistics
        mean = orig_data.mean()
        std = orig_data.std()
        min_val = orig_data.min()
        max_val = orig_data.max()

        # Special handling for 'brands' column - integer values with specific distribution
        if col == 'brands':
            # Generate brands with skewed distribution toward lower values (like original)
            # Use a combination of distributions to match original pattern
            values = np.zeros(num_samples)

            # 60% of samples: lower range (1-20) with clustering around small values
            low_count = int(num_samples * 0.6)
            # Use exponential distribution for natural skew toward lower values
            values[:low_count] = np.random.exponential(scale=5, size=low_count) + 1
            values[:low_count] = np.clip(values[:low_count], 1, 20)

            # 30% of samples: medium range (20-60)
            med_count = int(num_samples * 0.3)
            values[low_count:low_count+med_count] = np.random.uniform(20, 60, med_count)

            # 10% of samples: high range (60-130) for outliers
            high_count = num_samples - low_count - med_count
            values[low_count+med_count:] = np.random.uniform(60, 130, high_count)

            # Round to integers and add some randomization
            values = np.round(values).astype(int)

            # Add some variability by shuffling
            np.random.shuffle(values)

            # Match original NaN pattern
            nan_ratio = original_df[col].isna().sum() / len(original_df)
            num_nans = int(num_samples * nan_ratio)
            if num_nans > 0:
                nan_indices = np.random.choice(num_samples, num_nans, replace=False)
                values = values.astype(float)
                values[nan_indices] = np.nan

            synthetic_df[col] = values
            continue

        # Generate synthetic values with wider distribution for better class separation
        if std > 0:
            # Use larger standard deviation to create more spread
            values = np.random.normal(mean, std * 1.5, num_samples)
            # Allow values to extend beyond original range for better diversity
            values = np.clip(values, min_val * 0.5, max_val * 1.5)
        else:
            values = np.full(num_samples, mean)

        if col == 'RevenuesGhgCo':
            # Keep revenues relatively stable across all samples
            # This ensures the ESG score variation comes primarily from emissions

            # Apply industry-based revenue multipliers with less variation
            for i in range(num_samples):
                industry = synthetic_df.loc[i, 'SasbIndustry']
                if industry in industry_revenue_multipliers:
                    values[i] *= industry_revenue_multipliers[industry][i]
                else:
                    values[i] *= default_revenue_mult[i]

            # Apply moderate revenue variation
            revenue_bracket_multipliers = np.ones(num_samples)
            for i in range(num_samples):
                # Use consistent revenue multipliers to avoid extreme ratios
                revenue_bracket_multipliers[i] = np.random.uniform(0.8, 1.2)

            values *= revenue_bracket_multipliers

            # Ensure values stay within reasonable bounds
            values = np.clip(values, min_val * 0.5, max_val * 2.0)

        # Add some NaN values to match original sparsity
        nan_ratio = original_df[col].isna().sum() / len(original_df)
        num_nans = int(num_samples * nan_ratio)
        if num_nans > 0:
            nan_indices = np.random.choice(num_samples, num_nans, replace=False)
            values = values.astype(float)
            values[nan_indices] = np.nan

        synthetic_df[col] = values

    # NOW generate Scope1+2Total AFTER RevenuesGhgCo exists
    # Create 6 distinct emission/revenue ratio ranges for clean class separation
    if 'Scope1+2Total' in numeric_cols and 'RevenuesGhgCo' in synthetic_df.columns:
        samples_per_class = num_samples // 6
        emissions = np.zeros(num_samples)

        for i in range(num_samples):
            revenue = synthetic_df.loc[i, 'RevenuesGhgCo']
            if pd.isna(revenue) or revenue <= 0:
                emissions[i] = np.nan
                continue

            # Determine which risk class this sample belongs to
            class_idx = min(i // samples_per_class, 5)

            # Generate emission/revenue ratios in DISTINCT,  NON-OVERLAPPING ranges
            # These ranges are designed to create clear separation after log transformation
            if class_idx == 0:  # LOW risk - ratio: 0.01 to 1
                ratio = np.random.uniform(0.01, 1.0)
            elif class_idx == 1:  # MEDIUM_LOW risk - ratio: 1 to 10
                ratio = np.random.uniform(1.0, 10.0)
            elif class_idx == 2:  # MEDIUM risk - ratio: 10 to 100
                ratio = np.random.uniform(10.0, 100.0)
            elif class_idx == 3:  # MEDIUM_HIGH risk - ratio: 100 to 1,000
                ratio = np.random.uniform(100.0, 1000.0)
            elif class_idx == 4:  # HIGH risk - ratio: 1,000 to 10,000
                ratio = np.random.uniform(1000.0, 10000.0)
            else:  # VERY HIGH risk - ratio: 10,000 to 100,000
                ratio = np.random.uniform(10000.0, 100000.0)

            emissions[i] = revenue * ratio

        synthetic_df['Scope1+2Total'] = emissions

    # Generate categorical/text columns

    # Company Foreign (yes/no)
    foreign_ratio = (original_df['Company Foreign'] == 'yes').sum() / len(original_df)
    synthetic_df['Company Foreign'] = np.where(
        np.random.rand(num_samples) < foreign_ratio, 'yes', 'no'
    )

    # Public (yes/no)
    public_ratio = (original_df['Public'] == 'yes').sum() / len(original_df)
    synthetic_df['Public'] = np.where(
        np.random.rand(num_samples) < public_ratio, 'yes', 'no'
    )

    # Scope1+2Reported (yes/empty)
    reported_ratio = (original_df['Scope1+2Reported'] == 'yes').sum() / len(original_df)
    synthetic_df['Scope1+2Reported'] = np.where(
        np.random.rand(num_samples) < reported_ratio, 'yes', None
    )

    # RealZero (very rare in original)
    realzero_ratio = original_df['RealZero'].notna().sum() / len(original_df)
    synthetic_df['RealZero'] = np.where(
        np.random.rand(num_samples) < realzero_ratio, 'yes', None
    )

    # Generate dates (2020-2024 range)
    years = np.random.choice([2020, 2021, 2022, 2023, 2024], num_samples)
    months = np.random.choice(range(1, 13), num_samples)
    days = np.random.choice([28, 30, 31], num_samples)  # Simplified

    synthetic_df['DateEndFinancial2020'] = [
        f"{m}/{d}/2020" if y >= 2020 else np.nan
        for y, m, d in zip(years, months, days)
    ]
    synthetic_df['DateEndFinancial2021'] = [
        f"{m}/{d}/2021" if y >= 2021 else np.nan
        for y, m, d in zip(years, months, days)
    ]

    # GHG Report related columns
    report_types = ['CDP Report', 'Sustainability Report', 'ESG Report',
                    'Annual Report', 'Assurance Statement']
    synthetic_df['GhgReportName'] = np.random.choice(
        report_types + [np.nan],
        num_samples,
        p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2]
    )

    # Website (synthetic URLs)
    synthetic_df['Website'] = [
        f"www.synthetic-company-{i+1}.com#http://www.synthetic-company-{i+1}.com#"
        for i in range(num_samples)
    ]

    # YearGhgData
    synthetic_df['YearGhgData'] = np.random.choice(
        [2020, 2021, 2022, 2023, 2024, np.nan],
        num_samples,
        p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05]
    )

    # Fill in remaining columns from original with NaN
    for col in original_df.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = np.nan

    # Reorder columns to match original
    synthetic_df = synthetic_df[original_df.columns]

    return synthetic_df

def create_copies_of_original(original_df, num_copies=4):
    """
    Create copies of the original dataset with COPY: prefix.

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataset
    num_copies : int
        Number of copies to create (default: 4)

    Returns:
    --------
    list of pd.DataFrame
        List containing the copies
    """
    copies = []
    for i in range(num_copies):
        copy_df = original_df.copy()
        # Add COPY: prefix to company names
        copy_df['Company GHG Name'] = 'COPY: ' + copy_df['Company GHG Name'].astype(str)
        copies.append(copy_df)
    return copies

def combine_datasets(original_df, copies, synthetic_df):
    """
    Combine original, copies, and synthetic data, spreading copies throughout.

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataset (no prefix)
    copies : list of pd.DataFrame
        List of copied datasets (COPY: prefix)
    synthetic_df : pd.DataFrame
        Synthetic dataset (SYNTHETIC: prefix)

    Returns:
    --------
    pd.DataFrame
        Combined dataset with copies spread throughout
    """
    # Create list of all dataframes: original + 4 copies + synthetic
    all_dfs = [original_df] + copies + [synthetic_df]

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Shuffle the rows to spread everything throughout
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Regenerate ID column to be sequential
    combined_df['ID'] = range(1, len(combined_df) + 1)

    return combined_df

def save_synthetic_data(combined_df, output_path='synthetic_ghg_data.csv'):
    """Save combined data to CSV file."""
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Combined data saved to: {output_path}")

    # Count each type
    num_original = (combined_df['Company GHG Name'].str.startswith('COPY:') == False) & \
                   (combined_df['Company GHG Name'].str.startswith('SYNTHETIC:') == False)
    num_copies = combined_df['Company GHG Name'].str.startswith('COPY:').sum()
    num_synthetic = combined_df['Company GHG Name'].str.startswith('SYNTHETIC:').sum()

    print(f"\nDataset composition:")
    print(f"  - Original records: {num_original.sum()}")
    print(f"  - Copy records (COPY:): {num_copies}")
    print(f"  - Synthetic records (SYNTHETIC:): {num_synthetic}")
    print(f"  - Total records: {len(combined_df)}")

    print(f"\nFirst 10 company names (showing mix):")
    print(combined_df['Company GHG Name'].head(10))
    print(f"\nDataset shape: {combined_df.shape}")
    print(f"\nColumn dtypes:")
    print(combined_df.dtypes.value_counts())

def parse_arguments():
    """Parse command-line arguments."""
    num_samples = 1000  # Default value

    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
            if num_samples <= 0:
                print("Error: Number of samples must be positive")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid number '{sys.argv[1]}'. Must be a positive integer.")
            print("\nUsage: python generate_synthetic_data.py [num_samples]")
            print("Example: python generate_synthetic_data.py 1000")
            sys.exit(1)

    return num_samples

def main():
    """Main execution function."""
    print("="*80)
    print("ESG Synthetic Data Generator")
    print("="*80)

    # Parse command-line arguments
    num_samples = parse_arguments()
    print(f"\nGenerating {num_samples} synthetic samples...")

    # Load original data
    print("\n1. Loading original data...")
    original_df = load_original_data('ghg_data.csv')
    num_original = original_df.shape[0]
    print(f"   Original data: {num_original} rows, {original_df.shape[1]} columns")

    # Create 1 copy of original data (reduced from 4 to avoid clustering)
    print("\n2. Creating 1 copy of original data with COPY: prefix...")
    copies = create_copies_of_original(original_df, num_copies=1)
    print(f"   Created {len(copies)} copy, each with {num_original} rows")

    # Generate synthetic data
    print(f"\n3. Generating {num_samples} synthetic records with SYNTHETIC: prefix...")
    synthetic_df = generate_synthetic_data(original_df, num_samples=num_samples)

    # Combine all datasets
    print("\n4. Combining all datasets and shuffling...")
    combined_df = combine_datasets(original_df, copies, synthetic_df)
    total_rows = num_original + (1 * num_original) + num_samples
    print(f"   Total rows: {num_original} (original) + {1 * num_original} (copies) + {num_samples} (synthetic) = {total_rows}")

    # Save combined data
    print("\n5. Saving combined data...")
    save_synthetic_data(combined_df, output_path='synthetic_ghg_data.csv')

    # Display summary statistics comparison
    print("\n6. Comparison of key metrics:")
    print("="*80)

    key_cols = ['RevenuesGhgCo', 'Scope1+2Total', 'Scope1Emit', 'Scope2Total']
    for col in key_cols:
        if col in original_df.columns and col in synthetic_df.columns:
            orig_mean = original_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            combined_mean = combined_df[col].mean()
            print(f"{col:20s}: Original={orig_mean:>15.2f}, Synthetic={synth_mean:>15.2f}, Combined={combined_mean:>15.2f}")

    print("\n" + "="*80)
    print(f"DONE! Combined dataset saved to 'synthetic_ghg_data.csv'")
    print(f"Total records: {len(combined_df)}")
    print("="*80)

if __name__ == "__main__":
    main()
