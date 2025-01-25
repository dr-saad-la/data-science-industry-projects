"""
    Pima Indian Diabetes Project:
    Utility functions.
"""

__autho__ = "Dr. Saad Laouadi"
__email__ = "dr.saad.laouadi@gmail.com"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_diabetes_data():
    """Load and prepare the diabetes dataset"""
    url = (
        f"https://raw.githubusercontent.com/qcversity/ml-datasets/"
        f"refs/heads/main/data/pima_indians_diabetes.csv"
    )
    
    df = pd.read_csv(url,
                     names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                            'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'])
    return df
    


def create_age_groups(df):
    """Create age group categories"""
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 20, 30, 40, 50, 60, 100],
                           labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+'])
    return df


def create_bmi_categories(df):
    """Create BMI categories"""
    df['BMICategory'] = pd.cut(df['BMI'],
                              bins=[0, 18.5, 24.9, 29.9, 100],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return df

def calculate_risk_score(df):
    """Calculate patient risk scores"""
    high_glucose = df['Glucose'] > df['Glucose'].quantile(0.75)
    high_bmi = df['BMI'] > 30
    high_age = df['Age'] > 40
    family_history = df['DiabetesPedigree'] > df['DiabetesPedigree'].median()
    
    risk_factors = (high_glucose.astype(int) + 
                   high_bmi.astype(int) + 
                   high_age.astype(int) + 
                   family_history.astype(int))
    
    return risk_factors


def create_glucose_categories(df):
    """Create glucose control categories"""
    df['GlucoseControl'] = pd.cut(df['Glucose'],
                                 bins=[0, 70, 99, 125, 1000],
                                 labels=['Low', 'Normal', 'Pre-diabetic', 'Diabetic'])
    return df


def plot_age_distribution(df):
    """Plot age distribution by diabetes outcome"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y='Age', data=df)
    plt.title('Age Distribution by Diabetes Outcome')
    plt.show()

    
def plot_bmi_glucose_relationship(df):
    """Plot BMI vs Glucose with outcome"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='BMI', y='Glucose', hue='Outcome', alpha=0.6)
    plt.title('BMI vs Glucose by Diabetes Outcome')
    plt.show()

    
def generate_summary_statistics(df):
    """Generate comprehensive summary statistics"""
    summary = {
        'total_patients': len(df),
        'diabetes_rate': (df['Outcome'] == 1).mean() * 100,
        'avg_age': df['Age'].mean(),
        'avg_bmi': df['BMI'].mean(),
        'high_risk_count': len(df[df['RiskScore'] >= 3]),
        'normal_glucose_count': len(df[df['Glucose'] == 'Normal'])
    }
    return summary


def get_clinical_metrics_comparison(df):
    """Compare clinical metrics between diabetic and non-diabetic patients"""
    diabetic = df['Outcome'] == 1
    non_diabetic = ~diabetic
    
    metrics_comparison = pd.DataFrame({
        'Diabetic': df.loc[diabetic, ['Glucose', 'BMI', 'BloodPressure']].mean(),
        'Non-Diabetic': df.loc[non_diabetic, ['Glucose', 'BMI', 'BloodPressure']].mean(),
        'Difference_%': (df.loc[diabetic, ['Glucose', 'BMI', 'BloodPressure']].mean() / 
                        df.loc[non_diabetic, ['Glucose', 'BMI', 'BloodPressure']].mean() - 1) * 100
    }).round(2)
    
    return metrics_comparison