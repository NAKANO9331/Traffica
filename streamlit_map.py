import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pydeck as pdk

FLOWPATH = 'data/completed/traffic_result.h5'
adj_mat_path = 'data/raw/traffic/adj_mx.pkl'
graph_sensor_loc_path = 'data/raw/traffic/graph_sensor_locations.csv'

TOKEN = "YOUR_MAPBOX_TOKEN"

@st.cache_data(show_spinner=False)
def load_data():
    traff_flow_data = pd.read_hdf(FLOWPATH)
    traff_flow_data = traff_flow_data.iloc[:, :-2]
    sensor_loc_latlong = pd.read_csv(graph_sensor_loc_path)
    sensor_ref = pd.read_pickle(adj_mat_path)[0]
    sensor_ref_id_dict = pd.read_pickle(adj_mat_path)[1]
    r, c = traff_flow_data.shape
    traff_flow_data['date'] = pd.to_datetime(traff_flow_data.index)
    df_melted = pd.melt(traff_flow_data.reset_index(drop=True), id_vars="date", var_name='sensor_id', value_name='flow')
    df_melted['sensor_id'] = df_melted['sensor_id'].astype(int)
    new_data = pd.merge(df_melted, sensor_loc_latlong, on='sensor_id', how='outer')
    new_data.rename(columns={'index': 'sensor_index'}, inplace=True)
    new_data = new_data.sort_values(['sensor_index','date'], ignore_index=True)
    new_data['day'] = new_data['date'].dt.floor('D')
    mean_flow_df = new_data.groupby(['sensor_id', 'day']).agg({
        'flow': 'mean',  
        'sensor_index': 'first',
        'latitude': 'first',  
        'longitude': 'first',
    }).reset_index()
    mean_flow_df.rename(columns={'flow': 'mean_flow'}, inplace=True)
    mean_flow_df = mean_flow_df.sort_values(['day','sensor_index'], ignore_index=True)
    scaler = MinMaxScaler()
    mean_flow_df['normalized'] = scaler.fit_transform(mean_flow_df[['mean_flow']])
    mean_flow_df['inverse_normalized'] = mean_flow_df['normalized'].apply(lambda x: 1-x)
    levels = ["light", "medium", "heavy"]
    def choose_level(x):
        if x >= 55: return levels[0]
        elif x>=35 and x <55: return levels[1]
        elif x>0 and x <35: return levels[2]
        else: return levels[0]
    mean_flow_df['level'] = mean_flow_df['mean_flow'].apply(lambda x: choose_level(x))
    def get_radius(x):
        if x == "light": return 20
        elif x == 'medium': return 40
        else: return 80
    mean_flow_df['radius'] = mean_flow_df['level'].apply(lambda x: get_radius(x))
    return mean_flow_df

def plot_func(df_day, angle=0, mode="SUM"):
    # Text layer
    day_name = df_day.day.iloc[0].day_name()
    day = df_day.day.values[0]
    str_day = str(day).replace("T00:00:00.000000000", " ") + day_name
    text_data = [{"position": [-118.47605, 34.09478], "text": str_day},]
    # Layer separation
    light_df = df_day[df_day.level == "light"]
    medium_df = df_day[df_day.level == "medium"]
    heavy_df = df_day[df_day.level == "heavy"]
    # --- Bar chart data ---
    def expand_row(row):
        return pd.DataFrame([row] * int(100*row['inverse_normalized'])).assign(value=1)
    expanded_dfs = df_day.apply(expand_row, axis=1)
    grid_df = pd.concat(expanded_dfs.values.tolist()).reset_index(drop=True)
    light_grid = grid_df[(grid_df.level == "light")]
    medium_grid = grid_df[(grid_df.level == "medium") | (grid_df.level == "heavy")]
    heavy_grid = grid_df[(grid_df.level == "medium") | (grid_df.level == "heavy")]
    # Color scale
    COLOR_TECH_CYAN = [0, 255, 255]
    COLOR_TECH_BLUE = [102, 153, 255]
    COLOR_TECH_PURPLE = [204, 102, 255]
    COLOR_TECH_TEXT = [220, 220, 220, 255]
    COLOR_TECH_CYAN_SCALE = [
        [0, 255, 255],
        [51, 204, 255],
        [102, 255, 255],
        [153, 255, 255],
    ]
    COLOR_TECH_BLUE_SCALE = [
        [102, 153, 255],
        [51, 102, 204],
        [153, 204, 255],
        [102, 204, 255],
    ]
    COLOR_TECH_PURPLE_SCALE = [
        [204, 102, 255],
        [153, 51, 204],
        [229, 204, 255],
    ]
    # TextLayer
    text_layer = pdk.Layer(
        "TextLayer",
        text_data,
        get_position='position',
        get_text='text',
        get_size=20,
        get_color=COLOR_TECH_TEXT,
        get_angle=0,
        get_text_anchor="'middle'",
        get_alignment_baseline="'top'"
    )
    # HeatmapLayer
    light = pdk.Layer(
        "HeatmapLayer",
        data=light_df,
        opacity=1,
        get_position=["longitude", "latitude"],
        aggregation=pdk.types.String(mode),
        color_range=COLOR_TECH_CYAN_SCALE,
        threshold=1,
        get_weight="inverse_normalized",
        pickable=True,
    )
    medium = pdk.Layer(
        "HeatmapLayer",
        data=medium_df,
        opacity=1,
        get_position=["longitude", "latitude"],
        aggregation=pdk.types.String(mode),
        color_range=COLOR_TECH_BLUE_SCALE,
        threshold=1,
        get_weight="inverse_normalized",
        pickable=True,
    )
    heavy = pdk.Layer(
        "HeatmapLayer",
        data=heavy_df,
        opacity=1,
        get_position=["longitude", "latitude"],
        threshold=1,
        aggregation=pdk.types.String(mode),
        color_range=COLOR_TECH_PURPLE_SCALE,
        get_weight="radius",
        pickable=True,
    )
    # ScatterplotLayer
    light_scatter = pdk.Layer(
        "ScatterplotLayer",
        light_df,
        pickable=True,
        opacity=0.5,
        stroked=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        billboard = False,
        get_fill_color=COLOR_TECH_CYAN,
        get_line_color=COLOR_TECH_CYAN,
    )
    medium_scatter = pdk.Layer(
        "ScatterplotLayer",
        medium_df,
        pickable=True,
        opacity=0.5,
        stroked=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        billboard = False,
        get_fill_color=COLOR_TECH_BLUE,
        get_line_color=COLOR_TECH_BLUE,
    )
    heavy_scatter = pdk.Layer(
        "ScatterplotLayer",
        heavy_df,
        pickable=True,
        opacity=0.5,
        stroked=False,
        filled=True,
        radius_scale=6,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        billboard = False,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=COLOR_TECH_PURPLE,
        get_line_color=COLOR_TECH_PURPLE,
    )
    # Bar GridLayer
    medium_grid_layer = pdk.Layer(
        "GridLayer",
        medium_grid,
        pickable=True,
        extruded=True,
        cell_size=150,
        elevation_scale=4,
        elevationAggregation="MAX",
        get_position=["longitude", "latitude"],
        color_range=[COLOR_TECH_BLUE],
    )
    heavy_grid_layer = pdk.Layer(
        "GridLayer",
        heavy_grid,
        pickable=True,
        extruded=True,
        cell_size=150,
        elevation_scale=4,
        elevationAggregation="MAX",
        get_position=["longitude", "latitude"],
        color_range=[COLOR_TECH_PURPLE],
    )
    # View state
    view_state = pdk.ViewState(
        longitude=np.mean(df_day.longitude)- 0.035,
        latitude=np.mean(df_day.latitude)-0.015,
        zoom=10.5,
        pitch=55,
        bearing=angle
    )
    deck = pdk.Deck(
        height=600, width='100%',
        layers=[text_layer,light,medium,heavy,medium_grid_layer,heavy_grid_layer,light_scatter,medium_scatter,heavy_scatter],
        initial_view_state=view_state,
        map_style="dark",
        api_keys ={"mapbox":TOKEN},
        map_provider="mapbox",
    )
    return deck

# Streamlit interface
st.set_page_config(layout="wide")
st.title("🚦 Metr-LA Traffic Flow Prediction Interactive Map")

with st.spinner("Loading data..."):
    mean_flow_df = load_data()

all_days = mean_flow_df['day'].drop_duplicates().sort_values().tolist()
def day_to_str(d):
    return pd.to_datetime(d).strftime('%Y-%m-%d')

sel_day = st.sidebar.selectbox("Select Date", all_days, format_func=day_to_str)
df_day = mean_flow_df[mean_flow_df.day == sel_day]

st.subheader(f"Traffic Flow Map for {day_to_str(sel_day)}")
st.pydeck_chart(plot_func(df_day))