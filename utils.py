import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def normalize_to_range(arr, lower_bound=0, upper_bound=255):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Normalize to [0, 1]
    normalized = (arr - arr_min) / (arr_max - arr_min + 1e-7)
    
    # Scale to [lower_bound, upper_bound]
    scaled = normalized * (upper_bound - lower_bound) + lower_bound
    
    return scaled

def train_test_split_preserve_distr(df):
    # Define the Saffir-Simpson extended scale bins
    bins = [0, 33, 63, 82, 95, 112, 136, df['WMO_WIND'].max()]
    labels = ['Tropical Disturbance', 'Tropical Storm', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
    df['wind_category'] = pd.cut(df['WMO_WIND'], bins=bins, labels=labels, include_lowest=True)

    # Check to make sure we don't have any bins with less than 2 samples
    bin_counts = df['wind_category'].value_counts()
    if any(bin_counts < 2):
        raise ValueError('One of the categories has fewer than 2 samples, which is not suitable for stratified splitting.')

    # Perform the stratified split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['wind_category'])
    return train_df, test_df