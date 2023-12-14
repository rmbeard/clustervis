from typing import Any
import numpy as np
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import os
import pandas as pd
from pysal.lib import weights
from utils import aggregate_data, download_county_shapefile
import esda
from datetime import datetime
import folium
from folium import Choropleth, LayerControl, GeoJson
from branca.colormap import linear
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from splot.esda import moran_scatterplot
import plotly.graph_objs as go


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

def merge_data(gdf, df,date):
    #df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime if it's not already
    filtered_df = df[df['Date'].dt.date == date]  # Filter based on the selected date
    merged_gdf= gdf.merge(filtered_df[[st.session_state.column_selected, 'County_id', 'Date']], left_on='NAME', right_on='County_id', how='left')
    return merged_gdf

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
    #st.pydeck_chart(r)
    return r

def display_map_indicator(gdf, column_selected):
    # Handle NaN values
    gdf = gdf.dropna(subset=[column_selected])

    # Convert polygon geometries to centroid points
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Calculate a color scale
    min_val = gdf[column_selected].min()
    max_val = gdf[column_selected].max()

    def scale_color(value):
        if pd.isna(value):  # Check for NaN values
            return [0, 0, 0, 0]  # Transparent or some default color
        return [int(255 * (value - min_val) / (max_val - min_val)), 30, 0, 160]

    gdf['color'] = gdf[column_selected].apply(scale_color)

    # Set up the PyDeck Layer
    layer = pdk.Layer(
        'GeoJsonLayer',
        data=gdf.__geo_interface__,
        get_fill_color='color',
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
    return r

def cluster_analysis(merged_gdf, weight_method, column_selected):
    # Prepare the GeoDataFrame
   # gdf = gdf.dropna(subset=[column_selected,selected_date])  # Ensure there are no NaN values in the column of interest
    #print("cluster method testing", merged_gdf)
    # Determine the weights matrix based on the specified method
    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(merged_gdf)
    elif weight_method == 'IDW':
        # This is a placeholder - PySAL does not provide IDW weights for clustering
        # You would need to create or find a function to calculate these weights
        w = weights.distance.DistanceBand.from_dataframe(merged_gdf, threshold=1, binary=True)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(merged_gdf, threshold=1, binary=True)
    
    # Normalize the weights matrix
    #print(w)
    w.transform = 'r'
    
    # Perform Moran's I analysis for spatial autocorrelation
    y = merged_gdf[column_selected].values
    
    # Moran's I result is a single value indicating the overall spatial autocorrelation
    # For local indicators of spatial association, use Moran_Local
    local_moran = esda.Moran_Local(y, w)

    # Assuming local_moran is your Moran_Local object
    moran_df = pd.DataFrame({
    'Local_Moran_I': local_moran.Is,
    'Quadrant': local_moran.q,
    'P_Value': local_moran.p_sim,
    'Expected_I': local_moran.EI_sim,
    'Variance_I': local_moran.VI_sim,
    'StdDev_I': local_moran.seI_sim,
    'Z_Score_I': local_moran.z_sim
    })
    
    # to keep spatial information or other relevant data.
    # gdf is your original geodataframe
    gdf_moran = merged_gdf.join(moran_df)
    #print('moran analysis complete', gdf_moran)
    # For visualization, you can map the local Moran's I results, which indicate hot spots and cold spots
    # You might want to return the gdf with the new column or just the local_moran object, depending on your use case
    return gdf_moran

def cluster_analysis_and_plot(gdf, column_selected, weight_method):
    y = gdf[column_selected].values

    # Create weights matrix
    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(gdf)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(gdf, threshold=1, binary=True)
    else:
        raise ValueError(f"Weight method {weight_method} is not supported.")
    
    w.transform = 'r'
    y_lag = weights.lag_spatial(w, y)
    local_moran = esda.Moran_Local(y, w)

    # Join the local Moran's I statistics back to the original GeoDataFrame
    moran_df = pd.DataFrame({
    'Local_Moran_I': local_moran.Is,
    'Quadrant': local_moran.q,
    'P_Value': local_moran.p_sim,
    'Expected_I': local_moran.EI_sim,
    'Variance_I': local_moran.VI_sim,
    'StdDev_I': local_moran.seI_sim,
    'Z_Score_I': local_moran.z_sim
    })
     # gdf is your original geodataframe
    gdf_moran = gdf.join(moran_df)
    print('moran analysis complete', gdf_moran)
    # Create scatter plot
    #fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=y, y=y_lag, scatter=True, ci=None, ax=ax)
    #ax.axhline(0, color='grey', lw=1, linestyle='--')
    #ax.axvline(0, color='grey', lw=1, linestyle='--')
    #ax.set_xlabel(column_selected)
    #ax.set_ylabel("Spatial Lag")
    #ax.set_title("Moran's Scatter Plot")

    # Highlight significant values
    #significant = local_moran.p_sim < 0.05
    #ax.scatter(y[significant], local_moran.Is[significant], color='red')
    fig, ax = moran_scatterplot(moran, aspect_equal=True)
    plt.close(fig)  # Prevents the figure from being displayed immediately in Jupyter notebooks

    return gdf_moran, fig

def moran_quadrant_color(gdf, p_value_threshold=0.05):
    """
    Maps Moran's I quadrant to a specific color, taking into account the significance of the p-value.
    
    Parameters:
    - row: a row from the GeoDataFrame which includes 'Quadrant' and 'P_Value' columns.
    - p_value_threshold: the threshold below which the p-value is considered significant (default is 0.05).
    
    Returns:
    - A color string in hex format.
    """
    colors = {
        1: '#ff0000',  # Red
        2: '#00ffff',  # Light Blue
        3: '#0000ff',  # Blue
        4: '#ff69b4',  # Pink
        'insignificant': '#ffffff'  # White for insignificant p-values
    }
    
    if gdf['P_Value'] > p_value_threshold:
        return colors['insignificant']  # Use white if p-value is not significant
    
    return colors.get(gdf['Quadrant'], '#000000')  # Default color is black if no match

def display_hotspot_map_oldest(gdf):
    # Convert polygon geometries to centroid points
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    print('moran mapping', gdf)
    print(gdf.info())
    # Apply color mapping based on quadrant
    #gdf['color'] = gdf['quadrant'].map(quadrant_colors)
   # print(gdf['color'])
    # Assuming gdf is your GeoDataFrame and 'color' is the column with RGBA values
    #gdf['color'] = gdf[column_selected].apply(lambda x: scale_color(x))  # Replace wit h your scaling function
    # Assuming `gdf` is your GeoDataFrame and it has a column 'quadrant' with values 1, 2, 3, or 4
    gdf['color'] = gdf['Quadrant'].apply(moran_quadrant_color)
    print(gdf['color'].head())


    # Prepare GeoJson data for PyDeck
    geojson_data = gdf.__geo_interface__
    #print("conversion", gdf.__geo_interface__ )
    # Set up PyDeck layer
    layer = pdk.Layer(
        'GeoJsonLayer',
        data=geojson_data,
        #data=data,
        get_fill_color='color', # Set a static color for all features
        pickable=True,
        filled=True,
        #wireframe=True,
        #get_line_color=[0, 0, 0, 255],
        #line_width_min_pixels=1,
        auto_highlight=True
    )

    # Set the initial viewport for the map
    view_state = pdk.ViewState(
        longitude=gdf['lon'].mean(),
        latitude=gdf['lat'].mean(),
        zoom=6,
        pitch=0
    )

    # Create and return the PyDeck map
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.session_state.map=r
    return r

def display_hotspot_map_old(gdf):
    # Convert polygon geometries to centroid points for plotting points
    # If you want to plot polygons, you can skip this step
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Apply color mapping based on quadrant
    gdf['color'] = gdf['Quadrant'].apply(moran_quadrant_color)
    gdf=gdf.drop(columns=['Date'])
    
    # Create a folium map object
    m = folium.Map(location=[gdf['lat'].mean(), gdf['lon'].mean()], zoom_start=6)

    # Add the GeoJson overlay
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.8,
        },
       # highlight_function=lambda feature: {
       #     'weight': 3,
       #     'color': 'black',
        #},
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME', 'Local_Moran_I', 'Quadrant'],
            aliases=['County', 'Local Moran I', 'Quadrant'],
            localize=True
        ),
    ).add_to(m)
    st.session_state.map=m
    return m

def display_hotspot_map(gdf):
    # Check the unique values in 'Quadrant' to ensure they are what you expect
    print('Unique Quadrant values:', gdf['Quadrant'].unique())

    # Convert polygon geometries to centroid points for plotting points
    gdf['lon'] = gdf['geometry'].centroid.x
    gdf['lat'] = gdf['geometry'].centroid.y

    # Apply color mapping based on quadrant
    gdf['color'] = gdf.apply(moran_quadrant_color, axis=1)

    #print('Colors applied:', gdf['color'].head())  # Check the first few colors applied

    # Drop the 'Date' column if it exists
    if 'Date' in gdf.columns:
        gdf = gdf.drop(columns=['Date'])

    # Create a folium map object
    m = folium.Map(location=[gdf['lat'].mean(), gdf['lon'].mean()], zoom_start=6)

    # Add the GeoJson overlay
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.5,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME', 'Local_Moran_I', 'Quadrant'],
            aliases=['County', 'Local Moran I', 'Quadrant'],
            localize=True
        ),
    ).add_to(m)
    
    st.session_state.map=m
    return m


def plot_morans_scatter(gdf, column_selected, weight_method):
    y = gdf[column_selected].values

    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(gdf)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(gdf, threshold=1, binary=True)
    else:
        # Implement your custom weighting method here if needed
        w = None  

    if w is not None:
        w.transform = 'r'
        
        moran = esda.Moran_Local(y, w)

    

        # Plot
        fig, ax = moran_scatterplot(moran,  p=0.05)
        #sns.regplot(x=y, y=y_lag, ax=ax, scatter_kws={'s': 60}, line_kws={'color': 'blue'})
        #ax.axhline(0, color='grey', lw=1)
        #ax.axvline(0, color='grey', lw=1)
        ax.set_xlabel(column_selected)
        ax.set_ylabel("Spatial Lag")
        plt.close(fig)  # Prevents from displaying in a non-Streamlit environment

        return fig
    else:
        # Handle the case where the weight method is not implemented
        return None  


def plot_morans_scatter_new(gdf, column_selected, weight_method):
    y = gdf[column_selected].values
    w = None

    if weight_method == 'Queens Contiguity':
        w = weights.Queen.from_dataframe(gdf)
    elif weight_method == 'Distance Band':
        w = weights.DistanceBand.from_dataframe(gdf, threshold=1, binary=True)

    if w is not None:
        w.transform = 'r'

        # Calculate local Moran's I for the attribute
        local_moran_attribute = esda.Moran_Local(y, w)
        
        # Calculate local Moran's I for the spatial lag
        y_lag = weights.lag_spatial(w, y)
        local_moran_lag = esda.Moran_Local(y_lag, w)

        # Prepare data for the scatter plot
        scatter_data = pd.DataFrame({
            'Moran_Attribute': local_moran_attribute.Is,
            'Moran_Spatial_Lag': local_moran_lag.Is,
            'Significance': local_moran_attribute.p_sim < 0.05
        })

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='k')  # Assuming dark mode

        # Scatter points
        sns.scatterplot(
            data=scatter_data,
            x='Moran_Attribute',
            y='Moran_Spatial_Lag',
            hue='Significance',
            palette=['white', 'red'],  # Non-significant in white, significant in red
            legend=False,
            ax=ax
        )

        # Add trend line
        sns.regplot(
            data=scatter_data,
            x='Moran_Attribute',
            y='Moran_Spatial_Lag',
            scatter=False,
            color=text_color,
            ax=ax
        )

        # Customizing the plot for dark mode
        ax.figure.set_facecolor('k')
        ax.set_facecolor('k')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_title("Moran's I Scatterplot", color=text_color)

        # Annotations for quadrants
        ax.text(0.05, 0.95, 'HH', transform=ax.transAxes, color='white')
        ax.text(0.95, 0.95, 'HL', transform=ax.transAxes, color='white')
        ax.text(0.05, 0.05, 'LL', transform=ax.transAxes, color='white')
        ax.text(0.95, 0.05, 'LH', transform=ax.transAxes, color='white')

        # Adjust space for annotation outside the plot
        plt.subplots_adjust(bottom=0.2)

        # Add annotation text
        plt.annotate(
            'HH: High-High\nHL: High-Low\nLL: Low-Low\nLH: Low-High',
            xy=(0.5, -0.1),
            xycoords='axes fraction',
            ha='center',
            va='center',
            fontsize=12,
            color=text_color
        )

        plt.close(fig)  # Prevents from displaying in a non-Streamlit environment

        return fig
    else:
        return None


def plot_morans_i(columns_to_show, date_column='Date'):
       # Start the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the selected data
    for column in columns_to_show:
            sns.lineplot(data=column, x='Date', y=column, marker='o', label=column, ax=ax)
    
    # Enhance the plot
    plt.title('Comparison of Moran\'s I and P-values')
    plt.xlabel('Date')
    plt.ylabel(value_to_show)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    return fig


def calculate_cluster_frequencies(gdf, df, column_selected, weight_method):
    """
    Calculates the frequency of each cluster type for each unique date by performing local Moran's I analysis.

    Args:
    gdf (GeoDataFrame): GeoDataFrame containing the spatial data.
    df (DataFrame): DataFrame containing the temporal data with a 'Date' column.
    column_selected (str): The column to analyze.
    weight_method (str): The weight method to use.

    Returns:
    DataFrame: A DataFrame with cluster types as columns, dates as rows, and frequencies as values.
    """
    # Ensure the 'Date' column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Get unique dates from the df
    unique_dates = df['Date'].dt.date.unique()

    # Define a mapping from quadrant numbers to your column names
    quadrant_mapping = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}


    # Initialize a DataFrame to store frequencies
    cluster_freq = pd.DataFrame(columns=['HH', 'HL', 'LH', 'LL', 'ns'], index=unique_dates)

    for date in unique_dates:
        # Filter based on the selected date
        filtered_df = df[df['Date'].dt.date == date]
        # Merge filtered data with spatial data
        merged_gdf = gdf.merge(filtered_df[[column_selected, 'County_id']], left_on='NAME', right_on='County_id', how='left')
        
        # Call cluster_analysis function
        cluster_gdf = cluster_analysis(merged_gdf, weight_method, column_selected)
        
        # Filter based on significance
        significant_clusters = cluster_gdf[cluster_gdf['P_Value'] <= 0.05]
        insig_clusters = cluster_gdf[cluster_gdf['P_Value'] > 0.05]

      # Initialize the row with zeros
        cluster_freq.loc[date] = 0

        # Update the frequency of each significant cluster type
        freq_series = significant_clusters['Quadrant'].value_counts()
        #print('freq_series', freq_series)
        if not freq_series.empty:
            for quadrant_type, count in freq_series.items():
                # Use the mapping to update the correct column
                cluster_freq.at[date, quadrant_mapping[quadrant_type]] = count

                #print("q, count",quadrant_type, count)

        # Update the count of insignificant clusters
        cluster_freq.at[date, 'ns'] = insig_clusters.shape[0]

    #print('cluster_freq', cluster_freq)
    return cluster_freq


def create_frequency_table(cluster_freq_df):
    """
    Creates a formatted table of cluster frequencies.

    Args:
    cluster_freq_df (DataFrame): DataFrame from calculate_cluster_frequencies function.

    Returns:
    DataFrame: Formatted table suitable for inclusion in a research paper.
    """
    formatted_table = cluster_freq_df.copy()
    formatted_table['Total'] = formatted_table.sum(axis=1)
    return formatted_table


def plot_cluster_frequencies(cluster_freq_df):
    """
    Plots the frequency of each cluster type over time intervals.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set(style="whitegrid")
    cluster_freq_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('PLot timeseries')
    plt.xlabel('Time Interval')
    plt.ylabel('Frequency')
    plt.legend('cluster types')
    plt.close()
    return fig
    

def on_selection_change():
        print('current df', df)
        selected_date = st.session_state['selected_date']
        print(selected_date)
        column_selected = st.session_state['column_selected']
        weight_method = st.session_state['weight_method']
        print(column_selected)
        if isinstance(selected_date, str):
            # Parse the date string to a datetime.date object
            selected_date = datetime.strptime(selected_date, '%m/%d/%Y').date()
        
        filtered_df = df[df['Date'].dt.date == selected_date]  # Filter based on the selected date
        print("filtered data", filtered_df)
        merged_gdf= gdf.merge(filtered_df[[column_selected, 'County_id', 'Date']], left_on='NAME', right_on='County_id', how='left')
        
        
        print('merged data 2', merged_gdf)
      
        new_gdf=cluster_analysis(merged_gdf, weight_method, column_selected)
        st.session_state.merged_gdf=new_gdf
        map=display_hotspot_map(new_gdf)
        scatter = plot_morans_scatter(merged_gdf, st.session_state.column_selected, st.session_state.weight_method)
        # Assuming 'gdf' is your GeoDataFrame, 'date_column' is the name of your date column, and 'value_column' is the column you're analyzing
        # Example usage:
        #clust_freq_gdf=calculate_cluster_frequencies(gdf,df,st.session_state.slected_column, st.session_state.weight_method)
        
     
        #cluster_plot=plot_cluster_frequencies(clust_freq_gdf)
        st.session_state.scatter=scatter
        st.session_state.map=map
        st.session_state.merged_gdf=merged_gdf
        st.session_state.clust_freq_gdf=clust_freq_gdf
        st.session_state.cluster_plot=cluster_plot

def selection() -> None:
    # Interactive radio buttons
    option = st.sidebar.radio("Choose data input method:",
                      ('Use default data', 'Upload data (Coming Soon)'))
    #df=pd.Dataframe()
    # Conditional logic based on radio button choice
    if option == 'Use default data':
        st.sidebar.write("Using default data...")
        # Load and display your default data here
        df=st.session_state.df
        gdf=st.session_state.gdf

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
        
    # Add a dropdown for the weighting method
    st.sidebar.selectbox("Select Weighting Method", 
        ("Queens Contiguity", "IDW", "Distance Band"),
         key='weight_method',
        on_change=on_selection_change, # Call this function when value changes)
        #args=(column_selected,selected_date,weight_method)
    )

      # Use selectbox for column selection
    st.sidebar.selectbox( "Select column", 
        ("Total Cases Per 100k (7-day avg)", "Total Cases Per 100k"),
        key='column_selected',  # Link to session state
        on_change=on_selection_change # Call this function when value changes
    )

    time = st.sidebar.radio("Choose data aggregation over time:",
        ('Use default dates', 'Aggregate into time segments'))

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
        #st.session_state.df=df
    
    print("aggregated data",df)
    
    #Create Slider to select date
    df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime if it's not already
    # Convert Pandas Timestamp to datetime.date for the slider
    min_date = df['Date'].min().date()  # Convert to datetime.date
    max_date = df['Date'].max().date()  # Convert to datetime.date
    # Explanation text for Moran's I outputs

    st.markdown(explanation_text)
   # Create the slider to select the date
    st.slider(
        "Select a date to perform cluster detection:",
        min_value=min_date,
        max_value=max_date,
        value=min_date,
        format="MM/DD/YYYY",
        key='selected_date',
        on_change=on_selection_change  # Call the on_date_change function when the value changes
    )


   
st.set_page_config(page_title="Cluster Detection", page_icon="📹")
st.markdown("# Cluster Detection")
st.sidebar.header("Select inputs")
st.info('Case study of New York Counties: This dashboard visualizes variables of interest along side a demonstration of covid-19 geo-spatial clusters in US for a selected time')
#Add a checkbox in the sidebar to switch between light and dark mode
is_dark_theme = st.sidebar.checkbox('Dark theme', value=False)
# Choose color based on the theme
text_color = 'white' if is_dark_theme else 'black'
moran_i_info = (
"Moran's I is a measure of spatial autocorrelation that indicates how well the value "
"of a variable at one location is able to predict the value of that variable at nearby locations. "
"It ranges from -1 (indicating perfect dispersion) through 0 (perfect randomness) to +1 (perfect correlation)."
)
explanation_text = """
    **Explanation of Local Moran's I Spatial Autocorrelation Types:**

    - **High-High (HH):** An area with high values surrounded by areas with high values. Indicates a cluster of high values.
    - **Low-Low (LL):** An area with low values surrounded by areas with low values. Indicates a cluster of low values.
    - **Low-High (LH):** An area with low values surrounded by areas with high values. Indicates a potential outlier or transition zone.
    - **High-Low (HL):** An area with high values surrounded by areas with low values. Also indicates a potential outlier or transition zone.
    - **Not significant (ns):** Areas where the spatial pattern is less pronounced, and the local statistic is not significantly different from what would be expected in a random distribution.
    """
st.info(moran_i_info)
    
# Need to test out the method that chatgpt suggested
#def perform_cluster_analysis(gdf, weight_method):
# Initialize session state variables to their default values if they don't exist
if 'time_input' not in st.session_state:
   st.session_state['time_input'] = 'Use default data'

if 'weight_method' not in st.session_state:
    st.session_state['weight_method'] = 'Queens Contiguity'

weight_method = st.session_state.weight_method

if 'column_selected' not in st.session_state:
    st.session_state['column_selected'] = 'Total Cases Per 100k (7-day avg)'

column_selected = st.session_state.column_selected

if 'df' not in st.session_state:
    st.session_state['df'] = load_data()

df= st.session_state.df
df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime if it's not already

if 'gdf' not in st.session_state:
    st.session_state['gdf'] = load_shapefile()
    
gdf= st.session_state.gdf

if 'selected_date' not in st.session_state:
    st.session_state['selected_date'] =  df['Date'].min().date()

selected_date = df['Date'].min().date()  # Convert to datetime.date

if 'merged_gdf' not in st.session_state:
    st.session_state['merged_gdf'] =merge_data(gdf,df,selected_date)
 
merged_gdf = merge_data(gdf,df,selected_date)

if 'placeholder_map' not in st.session_state:
    st.session_state['placeholder_map']=cluster_analysis(st.session_state.merged_gdf, st.session_state.weight_method, st.session_state.column_selected)
   
if 'map' not in st.session_state:
    st.session_state['map'] =display_hotspot_map(st.session_state.placeholder_map)

if 'scatter' not in st.session_state:
    st.session_state['scatter']=plot_morans_scatter(merged_gdf, column_selected, weight_method)
    #st.session_state['scatter']=cluster_analysis_and_plot(merged_gdf, column_selected, weight_method)

#if ' clust_freq_gdf' not in st.session_state:
#    st.session_state[' clust_freq_gdf']=calculate_cluster_frequencies(gdf,df, st.session_state.column_selected, st.session_state.weight_method)
#clust_freq_gdf=st.session_state[' clust_freq_gdf']

if 'cluster_plot' not in st.session_state:
    st.session_state['cluster_plot']= plot_cluster_frequencies(clust_freq_gdf)


# Button to re-run the script
if st.sidebar.button("Reset"):
    st.session_state.df=df
    st.session_state.gdf=gdf
    st.session_state.selected_date=selected_date
    st.session_state.merged_gdf =merge_data(gdf,df,selected_date)
    st.session_state.placeholder_map=cluster_analysis(st.session_state.merged_gdf, st.session_state.weight_method, st.session_state.column_selected)
    #st.session_state.placeholder_map,st.session_state.scatter = cluster_analysis_and_plot(st.session_state.merged_gdf, st.session_state.weight_method, st.session_state.column_selected)
    st.session_state.map=display_hotspot_map(merged_gdf)
    st.session_state.scatter=plot_morans_scatter(merged_gdf, column_selected, weight_method)
    st.experimental_rerun()

#map = display_hotspot_map_testing(st.session_state.merged_gdf)
# PLace a base map here that can be used to display shapefiles
# To use this function in Streamlit
# gdf should be your joined GeoDataFrame with Moran's I results
selection()

#map=display_map(st.session_state.merged_gdf)
map = st.session_state.map
scatter=st.session_state.scatter
cluster_plot= st.session_state.cluster_plot
print('Current merged_gdf', st.session_state.merged_gdf.head(30))
folium_static(map)
st.write('Legend:')
# Define your legend style
legend_style = """
<style>
.dot {
  height: 15px;
  width: 15px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
}
</style>
"""

# Legend HTML
legend_html = f"""
{legend_style}
<div>
    <span class="dot" style="background-color: red;"></span> High-High (HH) <br>
    <span class="dot" style="background-color: pink;"></span> High-Low (HL) <br>
    <span class="dot" style="background-color: lightblue;"></span> Low-High (LH) <br>
    <span class="dot" style="background-color: blue;"></span> Low-Low (LL) <br>
    <span class="dot" style="background-color: grey;"></span> Not significant (ns)
</div>
"""


# Use the markdown function to render HTML
st.markdown(legend_html, unsafe_allow_html=True)

#st.plotly_chart(scatter)
st.pyplot(scatter)
st.sidebar.write('Choose comparative analysis')
#st.pyplot(cluster_plot)

print('done')
#st.pydeck_chart(map)




