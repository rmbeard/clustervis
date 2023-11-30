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
