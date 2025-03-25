#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
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
    
    # Add color control expander
    with st.expander("Line & Marker Colors", expanded=False):
        current_line_color = st.color_picker("Current Data Color", value="#0095FF")
        previous_line_color = st.color_picker("Previous Data Color", value="#9CA3AF")
        reference_line_color = st.color_picker("Reference Line Color", value="#10B981")
        shading_color = st.color_picker("Shading Color", value="#3B82F6")
        # Convert hex colors to rgba for shading with transparency
        r, g, b = int(shading_color[1:3], 16), int(shading_color[3:5], 16), int(shading_color[5:7], 16)
        shading_color_rgba = f"rgba({r}, {g}, {b}, 0.1)"
    
    show_reference = st.checkbox("Show Reference Lines", value=True)
    show_markers = st.checkbox("Show Data Points", value=True)
    show_shading = st.checkbox("Show Area Shading", value=True)
    marker_size = st.slider("Marker Size", 5, 15, 8)
    line_width = st.slider("Line Width", 1, 5, 2)
    annotation_size = st.slider("Annotation Text Size", 8, 20, 14)

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
    'current': current_line_color,  # Use color from picker
    'previous': previous_line_color,  # Use color from picker
    'reference': reference_line_color,  # Use color from picker
    'shading': shading_color_rgba  # Use rgba version of picked color
}

# Create the subplot without titles (we'll add them properly later)
fig = make_subplots(
    rows=2, 
    cols=1,
    vertical_spacing=0.25
)

def add_traces(df, row, measure_type):
    time = df['time'].values
    current_date = dates[0]
    mode = 'lines+markers+text' if show_markers else 'lines+text'
    
    # Get reference values
    reference_values = df['reference'].values
    current_values = df[current_date].values
    
    # Add current data
    fig.add_trace(
        go.Scatter(
            x=time,
            y=current_values,
            mode=mode,
            line=dict(color=colors['current'], width=line_width),
            marker=dict(
                size=marker_size,
                color=colors['current']
            ),
            text=current_values.round(1),
            textposition='top center',
            textfont=dict(
                family="Avenir",
                size=annotation_size,
                color="black"
            ),
            name=f"{current_date}",
            showlegend=(row == 1),  # Only show in legend for first plot
            legendgroup=current_date
        ),
        row=row, col=1
    )
    
    # Add shading between current line and reference where current > reference
    if show_shading:
        # For proper shading, we need to create fill areas only where current is above reference
        # We'll create a combined array with None values separating segments
        x_combined = []
        y_current = []
        y_reference = []
        
        for i in range(len(time)):
            if current_values[i] > reference_values[i]:
                # This point is above reference, include it
                x_combined.append(time[i])
                y_current.append(current_values[i])
                y_reference.append(reference_values[i])
                
                # If this is the last point or the next point is below reference,
                # we need to close the segment with None values
                if i == len(time) - 1 or current_values[i+1] <= reference_values[i+1]:
                    x_combined.append(None)
                    y_current.append(None)
                    y_reference.append(None)
            else:
                # If previous point was above reference, start a new segment
                if i > 0 and current_values[i-1] > reference_values[i-1]:
                    # We just crossed below, add the crossing point for a clean fill
                    # Find where the lines cross between this point and previous point
                    if current_values[i] != current_values[i-1]:  # Avoid division by zero
                        # Parametric value where lines cross
                        t = (reference_values[i-1] - current_values[i-1]) / (current_values[i] - current_values[i-1] - (reference_values[i] - reference_values[i-1]))
                        if 0 <= t <= 1:  # Valid crossing point
                            cross_x = time[i-1] + t * (time[i] - time[i-1])
                            cross_y = reference_values[i-1] + t * (reference_values[i] - reference_values[i-1])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
                            x_combined.append(None)
                            y_current.append(None)
                            y_reference.append(None)
                
                # If the next point will be above reference, start including points now
                if i < len(time) - 1 and current_values[i+1] > reference_values[i+1]:
                    # We're about to cross above, add the crossing point for a clean fill
                    if current_values[i+1] != current_values[i]:  # Avoid division by zero
                        # Parametric value where lines cross
                        t = (reference_values[i] - current_values[i]) / (current_values[i+1] - current_values[i] - (reference_values[i+1] - reference_values[i]))
                        if 0 <= t <= 1:  # Valid crossing point
                            cross_x = time[i] + t * (time[i+1] - time[i])
                            cross_y = reference_values[i] + t * (reference_values[i+1] - reference_values[i])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
        
        # Only add shading if we have points to shade
        if len(x_combined) > 0:
            # Add reference line as lower bound
            fig.add_trace(
                go.Scatter(
                    x=x_combined,
                    y=y_reference,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=1
            )
            
            # Add current line as upper bound with fill
            fig.add_trace(
                go.Scatter(
                    x=x_combined,
                    y=y_current,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=colors['shading'],
                    showlegend=False,
                    name='Above Reference',
                    hoverinfo='skip'
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
                    marker=dict(size=marker_size, color=colors['previous']),
                    text=df[prev_date].values.round(1),
                    textposition='top center',
                    textfont=dict(
                        family="Avenir",
                        size=annotation_size,
                        color="black"
                    ),
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
                marker=dict(size=marker_size, color=colors['reference']),
                text=df['reference'].values.round(1),
                textposition='top center',
                textfont=dict(
                    family="Avenir",
                    size=annotation_size,
                    color="black"
                ),
                showlegend=(row == 1),  # Only show reference in legend once
                legendgroup='reference'
            ),
            row=row, col=1
        )

# Add traces
add_traces(glucose_df, 1, "Glucose")
add_traces(insulin_df, 2, "Insulin")

    # Add proper subplot titles after figure creation
fig.update_layout(
    height=1200,  # Increased height
    showlegend=True,
    template='plotly_white',
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
    paper_bgcolor='rgba(0,0,0,0)',
    # Add proper titles for each subplot with adjusted positions
    annotations=[
        dict(
            text=f'Glucose Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.09,  # Moved down a bit
            showarrow=False
        ),
        dict(
            text=f'Insulin Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.48,  # Moved up to avoid overlap
            showarrow=False
        )
    ],
    legend=dict(
        yanchor="top",
        y=1.18,  # Moved higher up
        xanchor="center",
        x=0.5,
        font=dict(family="Avenir"),
        orientation="h"
    ),
    font=dict(family="Avenir"),
    margin=dict(t=200, r=50, b=50, l=50)  # Further increased top margin
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