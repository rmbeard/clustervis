from typing import Any
import numpy as np
import streamlit as st


def cluster() -> None:

    # Interactive radio buttons
    option = st.sidebar.radio("Choose data input method:",
                      ('Use default data', 'Upload covid data'))

    # Conditional logic based on radio button choice
    if option == 'Use default data':
        st.sidebar.write("Using default data...")
        # Load and display your default data here
        # default_data = load_default_data()
        st.write(default_data)
    elif option == 'Upload covid data':
        # File uploader widget
        uploaded_file = st.sidebar.file_uploader("Upload COVID data (comng soon)", type=['csv', 'xlsx'])
        # finish the coding needed to make this look right
        #st.sidebar.dropdown to select state
        if uploaded_file is not None:
            # Process the uploaded file
            # For example, if it's a CSV:
            # dataframe = pd.read_csv(uploaded_file)
            # st.write(dataframe)
            st.write("File uploaded successfully!")
    


    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    # Button to re-run the script
    if st.sidebar.button("Run"):
        st.experimental_rerun()

st.set_page_config(page_title="Animation Demo", page_icon="📹")
st.markdown("# Cluster Detection")
st.sidebar.header("Select inputs")
st.write(
    """This app visualizes variables of interest along side a demonstration of covid-19 geo-spatial clusters in US for a selected time""")
# PLace a base map here that can be used to display shapefiles



cluster()

