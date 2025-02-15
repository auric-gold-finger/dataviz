#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io

# At the top of the file, add these imports
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io

# Add a try-except block for kaleido
try:
    import kaleido
    KALEIDO_INSTALLED = True
except ImportError:
    KALEIDO_INSTALLED = False

# Page config
st.set_page_config(page_title="Glucose/Insulin Response", layout="wide")

# Custom CSS for fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&display=swap');
    
    .title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="title">Glucose and Insulin Response Analysis</p>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown('<p class="subtitle">Chart Controls</p>', unsafe_allow_html=True)
    
    # Move file uploader to sidebar
    uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'])
    
    show_reference = st.checkbox("Show Reference Lines", value=True)
    show_markers = st.checkbox("Show Data Points", value=True)
    show_shading = st.checkbox("Show Area Shading", value=True)
    marker_size = st.slider("Marker Size", 5, 15, 8)
    line_width = st.slider("Line Width", 1, 5, 2)

def process_data(df):
    # Rename columns appropriately
    df.columns = ['time', 'type', 'Sept 2022', 'Feb 2025', 'reference']
    dates = [col for col in df.columns if col not in ['time', 'type', 'reference']]
    dates.sort(reverse=True)
    glucose_df = df[df['type'] == 'glucose'].copy()
    insulin_df = df[df['type'] == 'insulin'].copy()
    return dates, glucose_df, insulin_df

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=0)
    st.session_state.df = df
elif st.session_state.df is None:
    # Sample data
    data = {
        'time': [0, 30, 60, 90] * 2,
        'type': ['glucose'] * 4 + ['insulin'] * 4,
        'Sept 2022': [103, 158, 142, 159] + [12, 72, 78, 107],
        'Feb 2025': [91, 148, 119, 85] + [8, 66, 75, 37],
        'reference': [90, 140, 120, 100] + [6, 40, 30, 20]
    }
    st.session_state.df = pd.DataFrame(data)

# Process data
dates, glucose_df, insulin_df = process_data(st.session_state.df)

# Colors
colors = {
    'above': 'rgb(239, 68, 68)',    # Red
    'below': 'rgb(59, 130, 246)',   # Blue
    'previous': 'rgb(156, 163, 175)', # Gray
    'reference': 'rgb(16, 185, 129)',  # Green
    'shading': 'rgba(59, 130, 246, 0.1)'  # Light blue
}

# Create figure
fig = make_subplots(
    rows=2, 
    cols=1,
    subplot_titles=(
        '<span style="font-family: Cormorant Garamond; font-size: 28px;">90 Minute Glucose Response to 75g Dextrose</span>',
        '<span style="font-family: Cormorant Garamond; font-size: 28px;">90 Minute Insulin Response to 75g dextrose</span>'
    ),
    vertical_spacing=0.25
)

def create_segment_colors(values, reference_values):
    """Create color array for line segments based on comparison with reference"""
    colors_list = []
    for i in range(len(values)-1):
        # If either point in the segment is above reference, color it red
        if values[i] > reference_values[i] or values[i+1] > reference_values[i+1]:
            colors_list.append(colors['above'])
        else:
            colors_list.append(colors['below'])
    return colors_list

def add_traces(df, row, measure_type):
    time = df['time'].values
    current_date = dates[0]
    mode = 'lines+markers+text' if show_markers else 'lines+text'
    
    # Get reference values for coloring
    reference_values = df['reference'].values
    current_values = df[current_date].values
    
    # Create segment colors
    segment_colors = create_segment_colors(current_values, reference_values)
    
    # Add current data with segments
    fig.add_trace(
        go.Scatter(
            x=time,
            y=current_values,
            mode=mode,
            line=dict(color=colors['above'], width=line_width),
            marker=dict(
                size=marker_size,
                color=[colors['above'] if v > r else colors['below'] 
                      for v, r in zip(current_values, reference_values)]
            ),
            text=current_values.round(1),
            textposition='top center',
            name=f"{current_date}",
            showlegend=(row == 1),  # Only show in legend for first plot
            legendgroup=current_date
        ),
        row=row, col=1
    )
    
    # Add shading
    if show_shading and len(dates) > 1:
        previous_values = df[dates[1]].values
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=current_values,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=previous_values,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=colors['shading'],
                showlegend=False
            ),
            row=row, col=1
        )
    
    # Add previous data
    if len(dates) > 1:
        for prev_date in dates[1:]:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=df[prev_date].values,
                    name=f"{prev_date}",
                    mode=mode,
                    line=dict(color=colors['previous'], width=line_width),
                    marker=dict(size=marker_size),
                    text=df[prev_date].values.round(1),
                    textposition='top center',
                    showlegend=(row == 1),  # Only show in legend for first plot
                    legendgroup=prev_date
                ),
                row=row, col=1
            )
    
    # Add reference line
    if show_reference:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=df['reference'].values,
                name="Reference",
                mode=mode,
                line=dict(color=colors['reference'], width=line_width, dash='dash'),
                marker=dict(size=marker_size),
                text=df['reference'].values.round(1),
                textposition='top center',
                showlegend=(row == 1),  # Only show reference in legend once
                legendgroup='reference'
            ),
            row=row, col=1
        )

# Add traces
add_traces(glucose_df, 1, "Glucose")
add_traces(insulin_df, 2, "Insulin")

fig.update_layout(
    height=1200,  # Increased height
    showlegend=True,
    template='plotly_white',
    title_font=dict(
        family="Cormorant Garamond",
        color="Black"
    ),
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="center",
        x=0.5,
        font=dict(family="Avenir"),
        orientation="h"
    ),
    font=dict(family="Avenir"),
    margin=dict(t=150, r=50, b=50, l=50)
)

# Update axes
fig.update_xaxes(
    title_text="Time (minutes)", 
    title_font=dict(family="Avenir"), 
    tickfont=dict(family="Avenir"),
    tickmode='array',
    tickvals=[0, 30, 60, 90]
)

fig.update_yaxes(
    title_text="Glucose (mg/dL)", 
    range=[80, 180],  # Increased upper limit to accommodate labels
    title_font=dict(family="Avenir"), 
    tickfont=dict(family="Avenir"), 
    row=1, 
    col=1
)
fig.update_yaxes(
    title_text="Insulin (µU/mL)", 
    range=[0, 120],  # Adjusted range for insulin plot
    title_font=dict(family="Avenir"), 
    tickfont=dict(family="Avenir"), 
    row=2, 
    col=1
)

# Display charts in a container
st.markdown('<p class="subtitle">Visualization</p>', unsafe_allow_html=True)
with st.container():
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button for PNG if kaleido is installed
    if KALEIDO_INSTALLED:
        if st.button("Download Plots as PNG"):
            # Convert plot to PNG
            img_bytes = fig.to_image(format="png", width=1200, height=1200, scale=2)
            
            # Create download button
            st.download_button(
                label="Click to Download PNG",
                data=img_bytes,
                file_name="glucose_insulin_response.png",
                mime="image/png"
            )
    else:
        st.info("To enable PNG download, install the kaleido package using: pip install kaleido")



# Metrics in containers
st.markdown('<p class="subtitle">Analysis</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    with st.container():
        current_date = dates[0]
        current_glucose = glucose_df[current_date].values
        peak_change = max(current_glucose) - current_glucose[0]
        if len(dates) > 1:
            previous_glucose = glucose_df[dates[1]].values
            previous_peak_change = max(previous_glucose) - previous_glucose[0]
            delta = peak_change - previous_peak_change
        else:
            reference_glucose = glucose_df['reference'].values
            previous_peak_change = max(reference_glucose) - reference_glucose[0]
            delta = peak_change - previous_peak_change
        st.metric(
            "Peak Glucose Change",
            f"{peak_change:.1f} mg/dL",
            f"{delta:.1f} mg/dL vs previous"
        )

with col2:
    with st.container():
        current_insulin = insulin_df[current_date].values
        peak_change = max(current_insulin) - current_insulin[0]
        if len(dates) > 1:
            previous_insulin = insulin_df[dates[1]].values
            previous_peak_change = max(previous_insulin) - previous_insulin[0]
            delta = peak_change - previous_peak_change
        else:
            reference_insulin = insulin_df['reference'].values
            previous_peak_change = max(reference_insulin) - reference_insulin[0]
            delta = peak_change - previous_peak_change
        st.metric(
            "Peak Insulin Change",
            f"{peak_change:.1f} µU/mL",
            f"{delta:.1f} µU/mL vs previous"
        )

# Data editor and download section at the bottom
st.markdown('<p class="subtitle">Data Editor</p>', unsafe_allow_html=True)
with st.container():
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True
    )

    # Download button for edited data
    csv = edited_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="edited_glucose_insulin_data.csv",
        mime="text/csv"
    )