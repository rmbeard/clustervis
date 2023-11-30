from typing import Any
import numpy as np
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import os
import pandas as pd
from pysal.lib import weights
from utils import aggregate_data, download_county_shapefile

# function to load case data on covid in New York, and display

# Define a function to reset the session states to their default values
def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]

def load_shapefile():
    # Define the path to the shapefile
    # If the shapefile is in a folder named 'shapefiles' at the root of your project and your app is also at the root
    shapefile_path = os.path.join('data', 'New York_shapefile.shp')  # Adjust this path as needed

    # Load shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    return gdf

def load_data():
    data_path = os.path.join('data', 'ny_covid_data.csv')  # Adjust this path as needed

    # Load file into a GeoDataFrame
    df = pd.read_csv(data_path)
    return df


def display_map_cluster(gdf, column):
    # Calculate a color scale from the column values, here we use a simple linear scale
    # For a more sophisticated scale, consider using branca.colormap or similar
    min_val = gdf[column].min()
    max_val = gdf[column].max()
    gdf['color'] = gdf[column].apply(lambda x: [int(255*(x-min_val)/(max_val-min_val)), 30, 0, 160])

    # Set up the PyDeck Layer
    layer = pdk.Layer(
        'GeoJsonLayer',
        data=gdf.__geo_interface__,
        get_fill_color='color',  # Use the 'color' column for fill color
        pickable=True,
        auto_highlight=True
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=gdf['lon'].mean(),
        latitude=gdf['lat'].mean(),
        zoom=6,
        pitch=0
    )
    
    # Render the map with PyDeck
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')
    st.pydeck_chart(r)

    # You can create a legend by displaying the color scale as a static image, or using Streamlit components to draw it
    # As an example, let's create a simple text-based legend
    st.write('Legend:')
    st.write(f'{min_val}: Red')
    st.write(f'{max_val}: Blue')


def display_map(gdf):
    # If your shapefile's geometry is polygons, convert it to centroids for point representation
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Set up the PyDeck Layer
    layer = pdk.Layer(
        'GeoJsonLayer',
        data=gdf.__geo_interface__,
        get_fill_color='[200, 30, 0, 160]',
        pickable=True
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=gdf['lon'].mean(),
        latitude=gdf['lat'].mean(),
        zoom=6,
        pitch=0
    )
    
    # Render the map with PyDeck
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

def selection() -> None:
    # Interactive radio buttons
    option = st.sidebar.radio("Choose data input method:",
                      ('Use default data', 'Upload data (Coming Soon)'))

    # Conditional logic based on radio button choice
    if option == 'Use default data':
        st.sidebar.write("Using default data...")
        # Load and display your default data here
        st.info('Case study of New York Counties')
        gdf=load_shapefile()
        df= load_data()
        default_display_data = 'Total Cases Per 100k (7-day avg)'
        display_map(gdf )
    elif option == 'Upload covid data':
        # File uploader widget
        uploaded_file = st.sidebar.file_uploader("Upload data (comng soon)", type=['csv', 'xlsx'])
        # finish the coding needed to make this look right
        #state = st.sidebar.selectbox("Select State", ("New York", "New Jersey", "California"))  # Add your own states
        #Census_Year= st.sidebar.selection('Select Year to down load census socioeconomic data',('2019','2020','2021'))
        if uploaded_file is not None:
            # Process the uploaded file
            # For example, if it's a CSV:
            # dataframe = pd.read_csv(uploaded_file)
            # st.write(dataframe)
            st.write("File uploaded successfully!")
            #time_segment = st.sidebar.selectbox("Select time segmentation present in the data",     
             #                            ("Daily", "Weekly", "Biweekly",'Monthly'))   
    
    # Add a dropdown for the weighting method
    weight_method = st.sidebar.selectbox("Select Weighting Method", 
                                         ("Queens Contiguity", "IDW", "Distance Band"))
    column_selected = st.sidebar.selectbox("Select column", 
                                         ("Total Cases Per 100k (7-day avg)", "Total Cases Per 100k"))
   
    time = st.sidebar.radio("Choose data aggregation over time:",
                      ('Use default data', 'Aggregate into time segments'))
    # Define a mapping of time input to frequency
    time_to_frequency = {
    'Weekly': 'W',
    'Biweekly': '2W',
    'Monthly': 'M'
}
    if time == "Aggregate into time segments":
        # Let the user select the time segmentation
        time = st.sidebar.selectbox("Select time segmentation", list(time_to_frequency.keys()))
        # Get the corresponding frequency from the dictionary
        freq = time_to_frequency.get(time, 'W')  # Default to 'W' if not found

        aggregated_data= aggregate_data(df,freq)
        df=aggregated_data
        print(df)
    # Button to perform cluster analysis
    if st.sidebar.button("Perform Cluster Analysis"):
        # Assuming 'gdf' is your GeoDataFrame containing the shapefile data
        merged_data = gdf.merge(df, left_on='NAME', right_on='County_id', how='left')
        result = perform_cluster_analysis(merged_data, weight_method, column_selected)
        # Display the analysis results
        # For example, if 'result' contains the centroids of clusters:
        display_map(result)

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    # Button to re-run the script
    if st.sidebar.button("Reset"):
        st.experimental_rerun()

def perform_cluster_analysis(gdf, weight_method):
    # Assume 'Total Cases Per 100k (7-day avg)' is already normalized and in the GeoDataFrame
    column_to_analyze = 'Total Cases Per 100k (7-day avg)'

    # Select the appropriate weighting method
    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(gdf)
    elif weight_method == 'IDW':
        # PySAL doesn't directly provide IDW for clustering, this is just a placeholder
        # You would need to create or find an appropriate function for this
        w = your_idw_function(gdf)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(gdf)

    # Normalize the weights
    w.transform = 'r'

    # Perform the cluster detection
    # This is a placeholder for where you would call the specific PySAL function you want to use
    # For example, for spatial autocorrelation analysis:
    # cluster_result = pysal.model.esda.Moran_Local(gdf[column_to_analyze], w)
    
    return cluster_result


# Need to test out the method that chatgpt suggested
#def perform_cluster_analysis(gdf, weight_method):

st.set_page_config(page_title="Animation Demo", page_icon="📹")
st.markdown("# Cluster Detection")
st.sidebar.header("Select inputs")
st.write(
    """This app visualizes variables of interest along side a demonstration of covid-19 geo-spatial clusters in US for a selected time""")
# PLace a base map here that can be used to display shapefiles

# Initialize session state variables to their default values if they don't exist
if 'time_input' not in st.session_state:
   st.session_state['time_input'] = 'Use default data'
if 'weight_method' not in st.session_state:
    st.session_state['weight_method'] = 'Queens Contiguity'
if 'column_selected' not in st.session_state:
    st.session_state['column_selected'] = 'Total Cases Per 100k (7-day avg)'

#function to make selections
selection()

