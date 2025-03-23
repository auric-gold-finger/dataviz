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
    # Get all column names except time, type, and reference
    dates = [col for col in df.columns if col not in ['time', 'type', 'reference']]
    
    # Parse dates to handle sorting
    def parse_date(date_str):
        months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        try:
            month_str, year_str = date_str.split()
            return int(year_str), months.get(month_str, 0)
        except:
            return (0, 0)  # Return tuple for proper sorting if parsing fails
    
    # Sort dates from newest to oldest
    dates.sort(key=parse_date, reverse=True)
    
    glucose_df = df[df['type'] == 'glucose'].copy()
    insulin_df = df[df['type'] == 'insulin'].copy()
    return dates, glucose_df, insulin_df

# Load data
# Initialize session state if 'df' doesn't exist
if 'df' not in st.session_state:
    # Sample data
    data = {
        'time': [0, 30, 60, 90] * 2,
        'type': ['glucose'] * 4 + ['insulin'] * 4,
        'Sept 2022': [103, 158, 142, 159] + [12, 72, 78, 107],
        'Feb 2025': [91, 148, 119, 85] + [8, 66, 75, 37],
        'reference': [90, 140, 120, 100] + [6, 40, 30, 20]
    }
    st.session_state.df = pd.DataFrame(data)

# Handle file upload
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=0)
    st.session_state.df = df

# Process data
dates, glucose_df, insulin_df = process_data(st.session_state.df)

# Colors
colors = {
    'above': 'rgb(0, 149, 255)',    # blue
    'below': 'rgb(0, 149, 255)',    # blue
    'previous': 'rgb(156, 163, 175)', # Gray
    'reference': 'rgb(16, 185, 129)',  # Green
    'shading': 'rgba(59, 130, 246, 0.1)'  # Light blue
}

# Update subplot titles using the dates
fig = make_subplots(
    rows=2, 
    cols=1,
    subplot_titles=(
        f'<span style="font-family: Cormorant Garamond; font-size: 28px;">Glucose Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})</span>',
        f'<span style="font-family: Cormorant Garamond; font-size: 28px;">Insulin Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})</span>'
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
    
    # Add conditional shading
    if show_shading:
        # Get comparison values (either previous test or reference)
        if len(dates) > 1:
            compare_values = df[dates[1]].values
            compare_label = dates[1]
        else:
            compare_values = df['reference'].values
            compare_label = 'reference'
        
        # Create arrays for selective shading
        x_shade = []
        y1_shade = []
        y2_shade = []
        
        for i in range(len(time)-1):
            # Add points for current segment
            if current_values[i] > compare_values[i] or current_values[i+1] > compare_values[i+1]:
                if i == 0 or len(x_shade) == 0:  # Start new section
                    x_shade.extend([time[i]])
                    y1_shade.extend([compare_values[i]])
                    y2_shade.extend([current_values[i]])
                
                x_shade.extend([time[i+1]])
                y1_shade.extend([compare_values[i+1]])
                y2_shade.extend([current_values[i+1]])
            else:
                if len(x_shade) > 0:  # End current section with None to break fill
                    x_shade.extend([None])
                    y1_shade.extend([None])
                    y2_shade.extend([None])
        
        if len(x_shade) > 0:
            # Add lower bound trace
            fig.add_trace(
                go.Scatter(
                    x=x_shade,
                    y=y1_shade,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Add upper bound trace with fill
            fig.add_trace(
                go.Scatter(
                    x=x_shade,
                    y=y2_shade,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=colors['shading'],
                    showlegend=False,
                    name=f'Exceeded {compare_label}'
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
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
    paper_bgcolor='rgba(0,0,0,0)', 
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
    tickvals=[0, 30, 60, 90, 120]
)

# Calculate y-axis ranges with padding
def get_axis_range(df, padding_percent=0.15):
    min_val = df[dates + ['reference']].min().min()
    max_val = df[dates + ['reference']].max().max()
    range_val = max_val - min_val
    padding = range_val * padding_percent
    return [min_val - padding, max_val + padding]

# Update y-axes with calculated ranges
glucose_range = get_axis_range(glucose_df)
insulin_range = get_axis_range(insulin_df)

fig.update_yaxes(
    title_text="Glucose (mg/dL)", 
    range=glucose_range,
    title_font=dict(family="Avenir"), 
    tickfont=dict(family="Avenir"), 
    row=1, 
    col=1
)
fig.update_yaxes(
    title_text="Insulin (µU/mL)", 
    range=insulin_range,
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

# Replace the existing Data Editor section with this enhanced version

st.markdown('<p class="subtitle">Data Management</p>', unsafe_allow_html=True)

# Add tabs for different data management operations
data_tab, column_tab = st.tabs(["Row Operations", "Column Management"])

with data_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced data editor with row operations
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",  # Allow adding/removing rows
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Add New Test Date")
        # Add new date column
        new_month = st.selectbox("Month", 
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        new_year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
        new_date = f"{new_month} {new_year}"
        
        if st.button("Add New Date Column"):
            if new_date not in edited_df.columns:
                # Add new column with default values
                glucose_default = edited_df[edited_df['type'] == 'glucose']['reference'].values
                insulin_default = edited_df[edited_df['type'] == 'insulin']['reference'].values
                defaults = pd.concat([pd.Series(glucose_default), pd.Series(insulin_default)])
                edited_df[new_date] = defaults
                st.session_state.df = edited_df
                st.success(f"Added new date column: {new_date}")
                st.rerun()
            else:
                st.error("This date column already exists!")

with column_tab:
    st.markdown("#### Manage Test Dates")
    
    # Get date columns (excluding special columns)
    date_columns = [col for col in edited_df.columns 
                   if col not in ['time', 'type', 'reference']]
    
    if date_columns:
        col_to_remove = st.selectbox("Select date to remove", date_columns)
        if st.button("Remove Selected Date"):
            edited_df = edited_df.drop(columns=[col_to_remove])
            st.session_state.df = edited_df
            st.success(f"Removed date column: {col_to_remove}")
            st.rerun()
    else:
        st.warning("No date columns to remove.")

# Validation and save changes
if st.button("Save Changes"):
    # Validate the data
    valid = True
    error_msg = []
    
    # Check for numeric values in date columns
    for col in [c for c in edited_df.columns if c not in ['time', 'type']]:
        if col != 'reference':
            try:
                pd.to_numeric(edited_df[col], errors='raise')
            except ValueError:
                valid = False
                error_msg.append(f"Non-numeric values found in column {col}")
    
    # Check for correct types
    if not all(t in ['glucose', 'insulin'] for t in edited_df['type']):
        valid = False
        error_msg.append("Invalid types found. Only 'glucose' and 'insulin' are allowed.")
    
    if valid:
        st.session_state.df = edited_df
        st.success("Changes saved successfully!")
    else:
        st.error("Validation failed:\n" + "\n".join(error_msg))

# Download button for edited data
csv = edited_df.to_csv(index=False)
st.download_button(
    label="Download Updated CSV",
    data=csv,
    file_name="edited_glucose_insulin_data.csv",
    mime="text/csv"
)