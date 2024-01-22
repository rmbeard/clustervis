import pandas as pd

def calculate_vaccine_protection(dataframe):
    # Ensure 'Date' is in datetime format
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe.fillna(0, inplace=True)
    # Define the key date thresholds
    primary_series_end_date = pd.to_datetime('2021-12-15')
    bivalent_start_date = pd.to_datetime('2022-11-16')
    
    # Initialize the new variable
    dataframe['Vaccine_Protection_Score'] = 0
    
    # Before the first key date, only primary series completion counts
    dataframe.loc[dataframe['Date'] < primary_series_end_date, 'Vaccine_Protection_Score'] = dataframe['Series_Complete_Pop_Pct']
    
    # After the first key date and before the second, both primary series and booster are equally important
    between_dates = (dataframe['Date'] >= primary_series_end_date) & (dataframe['Date'] < bivalent_start_date)
    dataframe.loc[between_dates, 'Vaccine_Protection_Score'] = (
        dataframe['Series_Complete_Pop_Pct'] + dataframe['Booster_Doses_50Plus_Vax_Pct']) / 2
    
    # After the second key date, primary series, booster, and bivalent booster are all equally important
    after_second_date = dataframe['Date'] >= bivalent_start_date
    dataframe.loc[after_second_date, 'Vaccine_Protection_Score'] = (
        dataframe['Series_Complete_Pop_Pct'] + dataframe['Booster_Doses_50Plus_Vax_Pct'] + dataframe.get('Bivalent_Booster_Pct', 0)) / 3
    
    # Normalize the score to be out of 100, if needed
    max_score = dataframe['Vaccine_Protection_Score'].max()
    if max_score > 0:  # Prevent division by zero
        dataframe['Vaccine_Protection_Score'] = dataframe['Vaccine_Protection_Score'] / max_score * 100
    
    return dataframe

def load_and_merge_datasets(vaccination_file_path, covid_data_file_path):
    # Load the datasets
    vaccinations_df = pd.read_csv(vaccination_file_path, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
    covid_data_df = pd.read_csv(covid_data_file_path, parse_dates=['Date'])
    
    # Renaming columns for consistency
    vaccinations_df.rename(columns={'Recip_County': 'County'}, inplace=True)

    # Removing "County" from the county names in the vaccination dataset
    vaccinations_df['County'] = vaccinations_df['County'].str.replace(' County', '', regex=False)

    # Merging the datasets on 'Date' and 'County' fields
    merged_df = pd.merge(covid_data_df, vaccinations_df, how='left', left_on=['Date', 'County_id'], right_on=['Date', 'County'])
     # Imputing missing values
    numerical_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for county in merged_df['County_id'].unique():
        county_data = merged_df[merged_df['County_id'] == county]
        merged_df.loc[merged_df['County_id'] == county, numerical_columns] = county_data[numerical_columns].interpolate(method='linear')

    return merged_df

# File paths (update these paths with the actual file locations)
vaccination_file_path = r"C:\Users\Beards\My Drive\ClusterView\my_spatial_genetics_project\data\raw_data\covid_data\New_York_vaccinations_by_county_9-17-21_5-10-23.csv"
covid_data_file_path = r"C:\Users\Beards\My Drive\ClusterView\my_spatial_genetics_project\data\raw_data\covid_data\ny_covid_data_aligned_shortened.csv"
# Load and merge the datasets
merged_data = load_and_merge_datasets(vaccination_file_path, covid_data_file_path)

# Calculate vaccine protection scores
merged_data = calculate_vaccine_protection(merged_data)

# Save the merged and calculated data to CSV files

merged_data.to_csv('testing_vaccine_composite.csv', index=False)

# Displaying the first few rows of the merged dataset
print(merged_data.head())

