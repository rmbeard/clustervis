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

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Cluster Vision",
        page_icon="👋",
    )

    st.write("# Cluster detection Analysis")

    st.sidebar.success("Select a task")

    st.markdown(
        """
        This web app demonstrates various interactive tasks for visual explorations to aid decision making in Public Health 
        """
    )

    st.info("Click on the left sidebar menu to navigate to the different visualization aids")
    st.subheader("Timelapse of early Covid -19 spread in US")
    st.markdown(
        """
        The following timelapse animation was produced using Google Earth Pro, to demonstrate the viral spread thought out the early pandemic according to phylogeographic analysis
    """
    ## possibly embed as a video instead gif
    ## <div style="padding: 56.25% 0 0 0; position: relative"><div style="height:100%;left:0;position:absolute;top:0;width:100%"><iframe height="100%" width="100%;" src="https://embed.wave.video/61b1016d46e0fb0001cf9ec3" frameborder="0" allow="autoplay; fullscreen" scrolling="no"></iframe></div></div>
    )

    #with row1_col2:
    #    st.image("https://github.com/giswqs/data/raw/main/timelapse/goes.gif")
    #    st.image("https://github.com/giswqs/data/raw/main/timelapse/fire.gif")
    st.image("https://github.com/rmbeard/data/raw/main/covid_vid_SparkVideo.gif",  width=350,)


if __name__ == "__main__":
    run()
