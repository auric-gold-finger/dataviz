#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Add a try-except block for kaleido
try:
    import kaleido
    KALEIDO_INSTALLED = True
except ImportError:
    KALEIDO_INSTALLED = False

# Page config
st.set_page_config(page_title="Glucose/Insulin Response", layout="wide")

# Title
st.title("Glucose and Insulin Response Analysis")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV data", type=['csv'])
    
    # Display options
    show_reference = st.checkbox("Show Reference Lines", value=True)
    show_markers = st.checkbox("Show Data Points", value=True)
    show_shading = st.checkbox("Show Area Shading", value=True)
    
    # Style controls
    current_line_color = st.color_picker("Current Data Color", value="#0095FF")
    previous_line_color = st.color_picker("Previous Data Color", value="#9CA3AF")
    reference_line_color = st.color_picker("Reference Line Color", value="#10B981")
    shading_color = st.color_picker("Shading Color", value="#3B82F6")
    
    # Convert hex to rgba for shading
    r, g, b = int(shading_color[1:3], 16), int(shading_color[3:5], 16), int(shading_color[5:7], 16)
    shading_color_rgba = f"rgba({r}, {g}, {b}, 0.1)"

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
            return (0, 0)
    
    # Sort dates from newest to oldest
    dates.sort(key=parse_date, reverse=True)
    
    glucose_df = df[df['type'] == 'glucose'].copy()
    insulin_df = df[df['type'] == 'insulin'].copy()
    return dates, glucose_df, insulin_df

def calculate_auc(time_points, values):
    """Calculate Area Under the Curve using trapezoidal rule"""
    if len(time_points) != len(values) or len(time_points) < 2:
        return 0
    
    auc = 0
    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i-1]
        avg_height = (values[i] + values[i-1]) / 2
        auc += dt * avg_height
    return auc

def calculate_matsuda_index(glucose_values, insulin_values):
    """Calculate Matsuda Index for insulin sensitivity"""
    try:
        fpg = glucose_values[0]
        fpi = insulin_values[0]
        mean_glucose = np.mean(glucose_values[1:]) if len(glucose_values) > 1 else glucose_values[0]
        mean_insulin = np.mean(insulin_values[1:]) if len(insulin_values) > 1 else insulin_values[0]
        matsuda = 10000 / np.sqrt((fpg * fpi) * (mean_glucose * mean_insulin))
        return matsuda
    except:
        return None

def calculate_homa_ir(glucose_fasting, insulin_fasting):
    """Calculate HOMA-IR"""
    try:
        return (glucose_fasting * insulin_fasting) / 405
    except:
        return None

# Load data
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
    'current': current_line_color,
    'previous': previous_line_color,
    'reference': reference_line_color,
    'shading': shading_color_rgba
}

# Create the subplot
fig = make_subplots(
    rows=2, 
    cols=1,
    vertical_spacing=0.15,
    subplot_titles=('Glucose Response', 'Insulin Response')
)

def add_traces(df, row, measure_type):
    time = df['time'].values
    current_date = dates[0]
    mode = 'lines+markers' if show_markers else 'lines'
    
    # Get reference values
    reference_values = df['reference'].values
    current_values = df[current_date].values
    
    # Add shading between current line and reference where current > reference
    if show_shading:
        # Create segments only where current > reference
        x_fill = []
        y_fill_upper = []
        y_fill_lower = []
        
        for i in range(len(time)):
            if current_values[i] > reference_values[i]:
                x_fill.append(time[i])
                y_fill_upper.append(current_values[i])
                y_fill_lower.append(reference_values[i])
            else:
                # Add None to break the fill when current <= reference
                if len(x_fill) > 0 and x_fill[-1] is not None:
                    x_fill.extend([None, None])
                    y_fill_upper.extend([None, None])
                    y_fill_lower.extend([None, None])
        
        # Only add shading if we have points where current > reference
        if len([x for x in x_fill if x is not None]) > 0:
            # Add lower boundary (reference line)
            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill_lower,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    name='fill_lower'
                ),
                row=row, col=1
            )
            
            # Add upper boundary (current line) with fill
            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill_upper,
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
    
    # Add current data
    fig.add_trace(
        go.Scatter(
            x=time,
            y=current_values,
            mode=mode,
            line=dict(color=colors['current'], width=2),
            marker=dict(size=6, color=colors['current']),
            name=f"{current_date}",
            showlegend=(row == 1),
            legendgroup=current_date
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
                    line=dict(color=colors['previous'], width=2),
                    marker=dict(size=6, color=colors['previous']),
                    showlegend=(row == 1),
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
                line=dict(color=colors['reference'], width=2, dash='dash'),
                marker=dict(size=6, color=colors['reference']),
                showlegend=(row == 1),
                legendgroup='reference'
            ),
            row=row, col=1
        )

# Add traces
add_traces(glucose_df, 1, "Glucose")
add_traces(insulin_df, 2, "Insulin")

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

# Update axes
fig.update_xaxes(title_text="Time (minutes)")
fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
fig.update_yaxes(title_text="Insulin (µU/mL)", row=2, col=1)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# Analysis Section
st.header("Analysis")

# Calculate metrics
current_date = dates[0]
time_points = glucose_df['time'].values
current_glucose = glucose_df[current_date].values
current_insulin = insulin_df[current_date].values

# Calculate indices
matsuda_index = calculate_matsuda_index(current_glucose, current_insulin)
homa_ir = calculate_homa_ir(current_glucose[0], current_insulin[0])
glucose_auc = calculate_auc(time_points, current_glucose)
insulin_auc = calculate_auc(time_points, current_insulin)

# Metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    peak_glucose = max(current_glucose) - current_glucose[0]
    st.metric("Peak Glucose Change", f"{peak_glucose:.1f} mg/dL")

with col2:
    peak_insulin = max(current_insulin) - current_insulin[0]
    st.metric("Peak Insulin Change", f"{peak_insulin:.1f} µU/mL")

with col3:
    if matsuda_index is not None:
        st.metric("Matsuda Index", f"{matsuda_index:.1f}")
    else:
        st.metric("Matsuda Index", "N/A")

with col4:
    if homa_ir is not None:
        st.metric("HOMA-IR", f"{homa_ir:.1f}")
    else:
        st.metric("HOMA-IR", "N/A")

# Interpretation
st.subheader("Interpretation")

interpretation_col1, interpretation_col2 = st.columns(2)

with interpretation_col1:
    st.markdown("**Glucose Response:**")
    baseline_glucose = current_glucose[0]
    peak_glucose_abs = max(current_glucose)
    
    if baseline_glucose > 126:
        st.write("Diabetic range - fasting glucose ≥126 mg/dL")
    elif baseline_glucose > 100:
        st.write("Impaired fasting glucose (100-125 mg/dL)")
    else:
        st.write("Normal fasting glucose (<100 mg/dL)")
    
    if len(current_glucose) >= 4:
        glucose_2h = current_glucose[-1]
        if glucose_2h >= 200:
            st.write("Diabetic range - 2h glucose ≥200 mg/dL")
        elif glucose_2h >= 140:
            st.write("Impaired glucose tolerance (140-199 mg/dL)")
        else:
            st.write("Normal glucose tolerance (<140 mg/dL)")

with interpretation_col2:
    st.markdown("**Insulin Response:**")
    baseline_insulin = current_insulin[0]
    
    if baseline_insulin > 25:
        st.write("Elevated fasting insulin - possible insulin resistance")
    elif baseline_insulin < 2:
        st.write("Low fasting insulin - possible β-cell dysfunction")
    else:
        st.write("Normal fasting insulin (2-25 µU/mL)")
    
    if matsuda_index is not None:
        if matsuda_index < 2.5:
            st.write(f"Low insulin sensitivity (Matsuda: {matsuda_index:.1f}) - insulin resistance")
        elif matsuda_index < 5.0:
            st.write(f"Reduced insulin sensitivity (Matsuda: {matsuda_index:.1f})")
        else:
            st.write(f"Good insulin sensitivity (Matsuda: {matsuda_index:.1f})")

# Data Management
st.header("Data Management")

# Data editor
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True
)

# Save and download
col1, col2 = st.columns(2)

with col1:
    if st.button("Save Changes"):
        st.session_state.df = edited_df
        st.success("Changes saved!")

with col2:
    csv = edited_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="glucose_insulin_data.csv",
        mime="text/csv"
    )

# Export plot
if KALEIDO_INSTALLED:
    if st.button("Download Plot as PNG"):
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        st.download_button(
            label="Click to Download PNG",
            data=img_bytes,
            file_name="glucose_insulin_response.png",
            mime="image/png"
        )
else:
    st.info("Install kaleido for image export: pip install kaleido")