"""
    Utility functions for bank marketing project. 
"""

__author__ = "Dr. Saad Laouadi"

import os
import zipfile
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

import pandas as pd
import numpy as np

import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def get_filename_from_url(url: str) -> str:
    """Extract filename from URL, handling encoded characters"""
    parsed_url = urlparse(url)
    filename = unquote(os.path.basename(parsed_url.path))
    return filename if filename else 'downloaded_file.zip'


def download_with_progress(url: str, save_path: str) -> str:
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
    
    return save_path


def extract_zip(zip_path: str, extract_path: str) -> bool:
    """Extract zip file contents"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all contents
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, desc="Extracting") as pbar:
                for member in zip_ref.namelist():
                    zip_ref.extract(member, extract_path)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")
        return False
    

class DataLoader:
    """Class for handling data loading and preprocessing for banking marketing dataset."""
    
    UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    
    @staticmethod
    def load_banking_data(data_path: str = 'data/banking',
                         file_name: str = 'bank-additional-full.csv',
                         force_download: bool = False) -> pd.DataFrame:
        """
        Load the banking marketing dataset, downloading it if necessary.
        
        Args:
            data_path: Directory where data should be stored
            file_name: Name of the specific CSV file to load
            force_download: If True, force a new download even if files exist
            
        Returns:
            pd.DataFrame: The loaded banking dataset
            
        Raises:
            FileNotFoundError: If the specified file is not found after download
            ValueError: If there are issues with data loading
        """
        # Create data directory if it doesn't exist
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we need to download the data
        file_path = data_path / file_name
        if not file_path.exists() or force_download:
            print(f"Downloading banking dataset from UCI repository...")
            
            # Download zip file
            zip_filename = get_filename_from_url(DataLoader.UCI_URL)
            temp_zip = data_path / zip_filename
            
            try:
                # Download the file
                download_with_progress(DataLoader.UCI_URL, temp_zip)
                
                # Extract contents
                print("\nExtracting files...")
                success = extract_zip(temp_zip, data_path)
                
                if not success:
                    raise ValueError("Failed to extract the dataset")
                
                # Remove zip file after extraction
                temp_zip.unlink()
                
                # Show extracted files
                print("\nExtracted files:")
                for f in data_path.glob('*'):
                    print(f"- {f.name}")
                    
            except Exception as e:
                raise ValueError(f"Failed to download and extract the dataset: {str(e)}")
        
        # Load the CSV file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file {file_name} not found in {data_path}. "
                f"Available files: {list(data_path.glob('*.csv'))}"
            )
            
        try:
            df = pd.read_csv(file_path, sep=';')
            print(f"\nSuccessfully loaded {file_name}")
            print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading the dataset: {str(e)}")


class BankingBaseAnalysis:
    """Base class for banking data preprocessing and common utilities."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with banking campaign dataset."""
        self.df = df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Create age groups
        self.df['age_group'] = pd.cut(
            self.df['age'], 
            bins=[0, 20, 30, 40, 50, 60, 100],
            labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
        )
        
        # Convert month to numeric
        month_map = {
            'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
            'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12
        }
        self.df['month_num'] = self.df['month'].map(month_map)
        
        # Add duration in minutes
        self.df['duration_min'] = self.df['duration'] / 60
        
        # Create contact frequency groups
        self.df['contact_group'] = pd.cut(
            self.df['campaign'], 
            bins=[-1, 0, 2, 5, float('inf')],
            labels=['No contact', '1-2 contacts', '3-5 contacts', '5+ contacts']
        )
        
        # Convert target to numeric for easier analysis
        self.df['target_numeric'] = (self.df['y'] == 'yes').astype(int)

        
class CustomerSegmentation(BankingBaseAnalysis):
    """Class for customer segmentation analysis in banking marketing campaigns."""

    def identify_high_potential_customers(self) -> pd.DataFrame:
        """
        Identify high-potential customers based on multiple criteria:
        - Age between 30-50
        - Higher education (university degree or professional course)
        - No defaults
        - Successfully contacted in previous campaigns
        """
        age_mask = self.df['age'].between(30, 50)
        education_mask = self.df['education'].isin(['university.degree', 'professional.course'])
        credit_mask = self.df['default'] == 'no'
        prev_success_mask = self.df['poutcome'] == 'success'
        
        high_potential = self.df[age_mask & education_mask & credit_mask]
        
        return high_potential


    def create_customer_profiles(self) -> pd.DataFrame:
        """
        Create detailed customer profiles using available characteristics:
        - Young Professionals: Young age, higher education, management/entrepreneur roles
        - Career Focused: Middle age, employed in technical or administrative roles
        - Retirees: Older age, retired
        - Students: Any age, student status
        """
        segments = {
            'Young Professionals': (
                (self.df['age'] < 35) & 
                (self.df['education'].isin(['university.degree', 'professional.course'])) &
                (self.df['job'].isin(['management', 'entrepreneur', 'self-employed']))
            ),
            'Career Focused': (
                (self.df['age'].between(35, 50)) & 
                (self.df['job'].isin(['admin.', 'technician', 'services', 'management']))
            ),
            'Retirees': (
                (self.df['age'] > 60) & 
                (self.df['job'] == 'retired')
            ),
            'Students': (
                self.df['job'] == 'student'
            )
        }
        
        profiles = {}
        for segment_name, mask in segments.items():
            segment_data = self.df[mask]
            
            # Calculate segment metrics
            profiles[segment_name] = {
                'count': len(segment_data),
                'success_rate': (segment_data['y'] == 'yes').mean() * 100,
                'avg_age': segment_data['age'].mean(),
                'avg_campaign_calls': segment_data['campaign'].mean(),
                'avg_call_duration': segment_data['duration_min'].mean(),
                'prev_success_rate': (segment_data['poutcome'] == 'success').mean() * 100
            }
        
        return pd.DataFrame(profiles).T.round(2)

    def analyze_education_impact(self) -> pd.DataFrame:
        """Analyze how education level impacts campaign success."""
        return self.analyze_success_by_group('education').sort_values('Success_Rate_%', ascending=False)
    
    def analyze_job_impact(self) -> pd.DataFrame:
        """Analyze how job type impacts campaign success."""
        return self.analyze_success_by_group('job').sort_values('Success_Rate_%', ascending=False)
    
    def analyze_financial_status(self) -> pd.DataFrame:
        """Analyze impact of financial status (loans, default history)."""
        # Combine housing and personal loan info
        self.df['loan_status'] = 'no_loans'
        self.df.loc[self.df['housing'] == 'yes', 'loan_status'] = 'housing_loan'
        self.df.loc[self.df['loan'] == 'yes', 'loan_status'] = 'personal_loan'
        self.df.loc[(self.df['housing'] == 'yes') & (self.df['loan'] == 'yes'), 'loan_status'] = 'both_loans'
        
        return self.analyze_success_by_group('loan_status')
    
    # def identify_high_potential_customers(self) -> pd.DataFrame:
    #     """Identify high-potential customers based on multiple criteria."""
    #     age_mask = self.df['age'].between(30, 50)
    #     balance_mask = self.df['balance'] > self.df['balance'].median()
    #     job_mask = self.df['job'].isin(['management', 'entrepreneur', 'self-employed'])
        
    #     return self.df[age_mask & balance_mask & job_mask]
    
    # def create_customer_profiles(self) -> pd.DataFrame:
    #     """Create detailed customer profiles using multiple characteristics."""
    #     segments = {
    #         'Young Professionals': (self.df['age'] < 35) & 
    #                              (self.df['job'].isin(['management', 'entrepreneur'])),
    #         'Stable Middle-Age': (self.df['age'].between(35, 50)) & 
    #                             (self.df['balance'] > 0),
    #         'Wealthy Seniors': (self.df['age'] > 50) & 
    #                          (self.df['balance'] > self.df['balance'].quantile(0.75)),
    #         'Students': self.df['job'] == 'student'
    #     }
        
    #     profiles = {}
    #     for segment_name, mask in segments.items():
    #         segment_data = self.df[mask]
            
    #         profiles[segment_name] = {
    #             'count': len(segment_data),
    #             'success_rate': (segment_data['y'] == 'yes').mean() * 100,
    #             'avg_balance': segment_data['balance'].mean(),
    #             'avg_campaign_calls': segment_data['campaign'].mean()
    #         }
        
    #     return pd.DataFrame(profiles).T.round(2)

    
class CampaignAnalysis(BankingBaseAnalysis):
    """Class for analyzing banking marketing campaign performance."""
    
    def analyze_success_by_group(self, column: str) -> pd.DataFrame:
        """Analyze campaign success rate by given column."""
        success_mask = self.df['y'] == 'yes'
        
        success_by_group = self.df[success_mask].groupby(column, observed=True)['y'].count()
        total_by_group = self.df.groupby(column, observed=True)['y'].count()
        
        success_rate = (success_by_group / total_by_group * 100).round(2)
        
        return pd.DataFrame({
            'Total_Customers': total_by_group,
            'Successful_Conversions': success_by_group,
            'Success_Rate_%': success_rate
        }).sort_values('Success_Rate_%', ascending=False)
    
    def analyze_campaign_timing(self) -> Tuple[pd.Series, pd.Series]:
        """Analyze campaign success rates by timing factors."""
        success_mask = self.df['y'] == 'yes'
        
        # Month analysis
        month_success = self.df[success_mask].groupby('month_num', observed=True)['y'].count()
        month_total = self.df.groupby('month_num', observed=True)['y'].count()
        month_rate = (month_success / month_total * 100).round(2)
        
        # Day analysis
        day_success = self.df[success_mask].groupby('day_of_week', observed=True)['y'].count()
        day_total = self.df.groupby('day_of_week', observed=True)['y'].count()
        day_rate = (day_success / day_total * 100).round(2)
        
        return month_rate, day_rate
    
    def analyze_contact_strategy(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze impact of contact strategy on campaign success."""
        # Contact frequency analysis
        contact_analysis = self.df.groupby('contact_group', observed=True).agg({
            'target_numeric': ['mean', 'count']
        })
        contact_analysis.columns = ['success_rate', 'total_contacts']
        contact_analysis['success_rate'] = contact_analysis['success_rate'] * 100
        
        # Previous outcome analysis
        prev_outcome_analysis = self.df.groupby('poutcome', observed=True).agg({
            'target_numeric': ['mean', 'count']
        })
        prev_outcome_analysis.columns = ['success_rate', 'total_contacts']
        prev_outcome_analysis['success_rate'] = prev_outcome_analysis['success_rate'] * 100
        
        return contact_analysis.round(2), prev_outcome_analysis.round(2)
    
    def calculate_efficiency_metrics(self) -> pd.DataFrame:
        """Calculate campaign efficiency metrics by month."""
        metrics = self.df.groupby('month', observed=True).agg({
            'duration_min': ['sum', 'mean'],
            'campaign': ['count', 'mean'],
            'target_numeric': 'mean'
        }).round(2)
        
        metrics.columns = [
            'total_duration_min', 'avg_duration_min',
            'total_contacts', 'avg_calls_per_customer',
            'success_rate'
        ]
        metrics['success_rate'] = metrics['success_rate'] * 100
        
        return metrics
    
    def generate_recommendations(self, customer_profiles: pd.DataFrame) -> List[str]:
        """Generate data-driven recommendations based on analysis."""
        recommendations = []
        
        # Age-based recommendations
        age_analysis = self.analyze_success_by_group('age_group')
        best_age_group = age_analysis.index[0]
        recommendations.append(
            f"Focus on {best_age_group} age group which shows highest success rate of "
            f"{age_analysis.loc[best_age_group, 'Success_Rate_%']}%"
        )
        
        # Timing recommendations
        month_rates, day_rates = self.analyze_campaign_timing()
        best_month = month_rates.idxmax()
        best_day = day_rates.idxmax()
        recommendations.append(
            f"Prioritize campaigns in month {best_month} and on {best_day} "
            f"with success rates of {month_rates[best_month]:.2f}% and "
            f"{day_rates[best_day]:.2f}% respectively"
        )
        
        # Contact strategy recommendations
        contact_analysis, _ = self.analyze_contact_strategy()
        best_contact_group = contact_analysis['success_rate'].idxmax()
        recommendations.append(
            f"Optimal contact frequency is {best_contact_group} "
            f"with success rate of {contact_analysis.loc[best_contact_group, 'success_rate']}%"
        )
        
        # Segment-specific recommendations
        best_segment = customer_profiles['success_rate'].idxmax()
        recommendations.append(
            f"Focus on {best_segment} segment with success rate of "
            f"{customer_profiles.loc[best_segment, 'success_rate']}%"
        )
        
        return recommendations        
            
    
class Visualization:
    """Class for visualizing banking marketing campaign insights."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with banking campaign dataset."""
        self.df = df.copy()
    
    def plot_key_insights(self):
        """Create visualizations of key campaign insights."""
        plt.figure(figsize=(15, 10))
        
        # Success rate by age group
        plt.subplot(2, 2, 1)
        age_success = self.df[self.df['y'] == 'yes'].groupby('age_group')['y'].count()
        age_total = self.df.groupby('age_group')['y'].count()
        age_rate = (age_success / age_total * 100)
        age_rate.plot(kind='bar')
        plt.title('Success Rate by Age Group')
        plt.ylabel('Success Rate (%)')
        
        # Success rate by month
        plt.subplot(2, 2, 2)
        month_success = self.df[self.df['y'] == 'yes'].groupby('month_num')['y'].count()
        month_total = self.df.groupby('month_num')['y'].count()
        month_rate = (month_success / month_total * 100)
        month_rate.plot(kind='line', marker='o')
        plt.title('Success Rate by Month')
        plt.ylabel('Success Rate (%)')
        
        # Success rate by job
        plt.subplot(2, 2, 3)
        job_success = self.df[self.df['y'] == 'yes'].groupby('job')['y'].count()
        job_total = self.df.groupby('job')['y'].count()
        job_rate = (job_success / job_total * 100)
        job_rate.sort_values(ascending=False).plot(kind='bar')
        plt.title('Success Rate by Job')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        
        # Balance distribution for successful vs unsuccessful
        plt.subplot(2, 2, 4)
        sns.boxplot(x='y', y='balance', data=self.df)
        plt.title('Balance Distribution by Outcome')
        
        plt.tight_layout()
        plt.show()