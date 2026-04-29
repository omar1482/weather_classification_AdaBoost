import pandas as pd
import sys

try:
    df = pd.read_csv('weather_classification_data.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    if 'Weather Type' in df.columns:
        print(f"\nWeather Type counts:\n{df['Weather Type'].value_counts()}")
    else:
        print("\nERROR: 'Weather Type' column missing.")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
