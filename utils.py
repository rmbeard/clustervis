# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
import pandas as pd
import streamlit as st
import os
import os
import re
import csv
import pandas as pd
from collections import defaultdict
import geopandas as gpd
import logging  
import requests
import pandas as pd
#from autocensus import Query
#from sodapy import Socrata
from datetime import datetime
import shutil
from zipfile import ZipFile

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


def aggregate_data(df, frequency):
    # Convert the 'Test Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values('Date')
    
    # Group by 'County_id' and then resample for each county
    # 'Grouper' allows you to specify a frequency for resampling and group at the same time
    aggregated_df = df.groupby(['County_id', pd.Grouper(key='Date', freq=frequency)]).sum()
    
    # Reset index to turn 'County_id' and 'Test Date' back into columns
    aggregated_df.reset_index(inplace=True)

    return aggregated_df

def download_county_shapefile(state_name, project_root):
    #logging.info("Downloading county shapefile.")  # Added logging
    # Use the project_root parameter to define the output directory and file path
    output_directory = os.path.join(project_root, "data", "shapefiles")
    output_file_path = os.path.join(output_directory, f"{state_name}_shapefile.shp")

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # This uses US Census Bureau's public shapefile repository for U.S. counties
    url = f"https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip"
    
    # Load the shapefile from the URL
    gdf = gpd.read_file(url)
    
    # Filter to only the specified state and save to the output path
    state_gdf = gdf[gdf['STATE_NAME'] == state_name]
    state_gdf.to_file(output_file_path)
    
    print(f"Shapefile for {state_name} saved to {output_file_path}")
    print("Shapefile downloaded successfully.")  # Added print statement
   # logging.info("Shapefile downloaded successfully.")  # Added logging
    return state_gdf

# api call to census data on social and demographic data
def fetch_and_normalize_census_data(api_key, state_code, all_years):
    # Variables to be downloaded
    variables = [
        'B01003_001E',  # Total population
        'B08006_008E',  # Public transportation usage (excluding taxicab)
        'B08006_001E',  # Total Public transportation respondents

        # Insurance coverage by age groups
        'B27010_001E',  # Total uninsured by age status known
        'B27010_017E',  # No_HI_coverage_under_19
        'B27010_033E',  # No_HI_coverage_19-34
        'B27010_050E',  # No_HI_coverage_35-64
        'B27010_066E',  # No_HI_coverage_65_up

        'C17002_001E',  # Total population for whom poverty status is determined
        'C17002_002E',  # Total population below the poverty line

        'B19056_002E',  # With public assistance income
        'B19056_001E',  # Total with public assistance income status known

        'B25070_010E',  # Renter occupied housing units spending more than 50% income on rent
        'B25070_001E',  # Total renter occupied housing units with cost burden status known

        'B03003_001E',  # Total population status Hispanic known
        'B03003_003E',  # Total population Hispanic or Latino
    ]

    query = Query(
        estimate=5,
        years=all_years,
        variables=variables,
        for_geo=['county:*'],
        in_geo=[f'state:{state_code}'],
        census_api_key=api_key
    )

    # Run query and collect output in dataframe
    df = query.run()
    pivot = df.pivot_table(columns= ['variable_code'],index=['name','geo_id','year'],values= 'value')
    df=pivot
    # Normalize the data per 100,000 people
    normalization_factor = 100000

    # Total uninsured is the sum of uninsured across all age groups
    df['Total_uninsured'] = df['B27010_017E'].astype(int) + df['B27010_033E'].astype(int) + \
                            df['B27010_050E'].astype(int) + df['B27010_066E'].astype(int)

    df['Public_transportation_per_100k'] = df['B08006_008E'].astype(int) / df['B08006_001E'].astype(int) * normalization_factor
    df['Uninsured_per_100k'] = df['Total_uninsured'] / df['B27010_001E'].astype(int) * normalization_factor
    df['Poverty_per_100k'] = df['C17002_002E'].astype(int) / df['C17002_001E'].astype(int) * normalization_factor
    df['Public_income_assistance_per_100k'] = df['B19056_002E'].astype(int) / df['B19056_001E'].astype(int) * normalization_factor
    df['Rent_greater_50_percent_income_per_100k'] = df['B25070_010E'].astype(int) / df['B25070_001E'].astype(int) * normalization_factor
    df['Hispanic_or_Latino_per_100k'] = df['B03003_003E'].astype(int) / df['B03003_001E'].astype(int) * normalization_factor
    
    columns_to_drop = [var for var in variables if var != 'B01003_001E']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # reset index, rename some columns for clarity
    df = df.reset_index()

    df = df.rename(columns={'B01003_001E': 'Tot_pop', 'name': 'County'})
    df['County'] = df['County'].str.replace(" County, New York", "")
    return df

if __name__ == "__main__":
    # Paths and parameters (could be adapted to be command-line arguments or environment variables for more flexibility)
    metadata_path = r"C:\Users\Beards\My Drive\ClusterView\my_spatial_genetics_project\data\raw_data\genetic_data\metadata_gisaid_10_2_2023.tsv"
    #location_data_path = r"C:\Users\Beards\My Drive\ClusterView\my_spatial_genetics_project\data\raw_data\location_data\covid_county_population_usafacts.csv"
    location_data_path = r"C:\Users\Beards\My Drive\ClusterView\my_spatial_genetics_project\data\raw_data\location_data\covid_county_population_usafacts.csv"
    # Set project root directly
    project_root = "C:\\Users\\Beards\\My Drive\\ClusterView\\my_spatial_genetics_project"
    state = "New York"  # Or any other state
    state_code= "NY"
    start_date = "2020-03-27"
    end_date = "2023-12-08"
    all_years= [2020,2021]
    api_key = '29409afe6a95cab8c573286e6d599effa54d21cf'
    df= fetch_and_normalize_census_data(api_key, state_code, all_years)
    print(df)
   

