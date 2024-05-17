import pandas as pd
import json
from sklearn.impute import SimpleImputer
import numpy as np

def parse_json_column(column):
    def parse_json(value):
        try:
            return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            return None
    return column.apply(parse_json)

# Read the CSV file, skipping the first row (header)
file_path = 'data.csv'
df = pd.read_csv(file_path, skiprows=1, header=None)

# Define the column names
df.columns = [
    '@id', '@type', 'elevation', 'station', 'timestamp', 'rawMessage', 'textDescription', 'icon',
    'presentWeather', 'temperature', 'dewpoint', 'windDirection', 'windSpeed', 'windGust',
    'barometricPressure', 'seaLevelPressure', 'visibility', 'maxTemperatureLast24Hours',
    'minTemperatureLast24Hours', 'precipitationLastHour', 'precipitationLast3Hours',
    'precipitationLast6Hours', 'relativeHumidity', 'windChill', 'heatIndex', 'cloudLayers'
]

# List of columns that contain JSON data
json_columns = [
    'elevation', 'temperature', 'dewpoint', 'windDirection', 'windSpeed', 'windGust', 
    'barometricPressure', 'seaLevelPressure', 'visibility', 'maxTemperatureLast24Hours', 
    'minTemperatureLast24Hours', 'precipitationLastHour', 'precipitationLast3Hours', 
    'precipitationLast6Hours', 'relativeHumidity', 'windChill', 'heatIndex', 'cloudLayers'
]

# Parsing the JSON columns
for col in json_columns:
    df[col] = parse_json_column(df[col])

# Extracting values from the parsed JSON
for col in json_columns:
    df[f'{col}_value'] = df[col].apply(lambda x: x['value'] if isinstance(x, dict) and 'value' in x else None)

# Drop the original JSON columns
df.drop(columns=json_columns, inplace=True)

# Selecting features and target variable (example target: temperature_value)
features = df.drop(columns=['temperature_value', '@id', '@type', 'station', 'timestamp', 'rawMessage', 'icon', 'presentWeather'])
target = df['temperature_value']

# Convert categorical variables if needed (e.g., 'textDescription')
features = pd.get_dummies(features, columns=['textDescription'], drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)
target = target.fillna(target.mean())

# Save processed data to disk
np.savez('processed_data.npz', X=features, y=target)
