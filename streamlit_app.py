#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
from datetime import datetime

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
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .interpretation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0095FF;
        margin: 1rem 0;
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
    
    # Add export options
    with st.expander("Export Options", expanded=False):
        export_width = st.slider("Export Width (px)", 800, 2000, 1200)
        export_height = st.slider("Export Height (px)", 600, 1600, 1200)
        export_scale = st.slider("Export Scale", 1, 4, 2)
        export_format = st.selectbox("Export Format", ["png", "jpeg", "svg", "pdf"])
    
    # Add color control expander
    with st.expander("Line & Marker Colors", expanded=False):
        current_line_color = st.color_picker("Current Data Color", value="#0095FF")
        previous_line_color = st.color_picker("Previous Data Color", value="#9CA3AF")
        reference_line_color = st.color_picker("Reference Line Color", value="#10B981")
        shading_color = st.color_picker("Shading Color", value="#3B82F6")
        # Convert hex colors to rgba for shading with transparency
        r, g, b = int(shading_color[1:3], 16), int(shading_color[3:5], 16), int(shading_color[5:7], 16)
        shading_color_rgba = f"rgba({r}, {g}, {b}, 0.1)"
    
    # Display options
    with st.expander("Display Options", expanded=True):
        show_reference = st.checkbox("Show Reference Lines", value=True)
        show_markers = st.checkbox("Show Data Points", value=True)
        show_shading = st.checkbox("Show Area Shading", value=True)
        show_interpretation = st.checkbox("Show Clinical Interpretation", value=True)
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

def interpret_glucose_response(glucose_values, time_points):
    """Provide clinical interpretation of glucose response"""
    baseline = glucose_values[0]
    peak = max(glucose_values)
    peak_time = time_points[np.argmax(glucose_values)]
    final = glucose_values[-1]
    
    interpretations = []
    
    # Baseline interpretation
    if baseline < 70:
        interpretations.append("⚠️ Low baseline glucose - possible hypoglycemia")
    elif baseline > 100:
        interpretations.append("⚠️ Elevated baseline glucose - possible impaired fasting glucose")
    else:
        interpretations.append("✅ Normal baseline glucose")
    
    # Peak response
    if peak > 200:
        interpretations.append("⚠️ High peak glucose response - possible glucose intolerance")
    elif peak > 140:
        interpretations.append("⚡ Moderate peak glucose response")
    else:
        interpretations.append("✅ Normal peak glucose response")
    
    # Recovery
    if final > baseline + 20:
        interpretations.append("⚠️ Glucose not returning to baseline - possible delayed clearance")
    else:
        interpretations.append("✅ Good glucose recovery")
    
    return interpretations

def interpret_insulin_response(insulin_values, time_points):
    """Provide clinical interpretation of insulin response"""
    baseline = insulin_values[0]
    peak = max(insulin_values)
    peak_time = time_points[np.argmax(insulin_values)]
    
    interpretations = []
    
    # Baseline interpretation
    if baseline > 15:
        interpretations.append("⚠️ Elevated baseline insulin - possible insulin resistance")
    else:
        interpretations.append("✅ Normal baseline insulin")
    
    # Peak response
    if peak > 100:
        interpretations.append("⚠️ High peak insulin response - possible insulin resistance")
    elif peak < 30:
        interpretations.append("⚠️ Low peak insulin response - possible impaired insulin secretion")
    else:
        interpretations.append("✅ Normal peak insulin response")
    
    # Response timing
    if peak_time <= 30:
        interpretations.append("⚡ Early peak insulin response")
    elif peak_time >= 90:
        interpretations.append("⚠️ Delayed peak insulin response")
    else:
        interpretations.append("✅ Normal timing of peak insulin response")
    
    return interpretations

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
            showlegend=(row == 1),
            legendgroup=current_date
        ),
        row=row, col=1
    )
    
    # Add shading between current line and reference where current > reference
    if show_shading:
        x_combined = []
        y_current = []
        y_reference = []
        
        for i in range(len(time)):
            if current_values[i] > reference_values[i]:
                x_combined.append(time[i])
                y_current.append(current_values[i])
                y_reference.append(reference_values[i])
                
                if i == len(time) - 1 or current_values[i+1] <= reference_values[i+1]:
                    x_combined.append(None)
                    y_current.append(None)
                    y_reference.append(None)
            else:
                if i > 0 and current_values[i-1] > reference_values[i-1]:
                    if current_values[i] != current_values[i-1]:
                        t = (reference_values[i-1] - current_values[i-1]) / (current_values[i] - current_values[i-1] - (reference_values[i] - reference_values[i-1]))
                        if 0 <= t <= 1:
                            cross_x = time[i-1] + t * (time[i] - time[i-1])
                            cross_y = reference_values[i-1] + t * (reference_values[i] - reference_values[i-1])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
                            x_combined.append(None)
                            y_current.append(None)
                            y_reference.append(None)
                
                if i < len(time) - 1 and current_values[i+1] > reference_values[i+1]:
                    if current_values[i+1] != current_values[i]:
                        t = (reference_values[i] - current_values[i]) / (current_values[i+1] - current_values[i] - (reference_values[i+1] - reference_values[i]))
                        if 0 <= t <= 1:
                            cross_x = time[i] + t * (time[i+1] - time[i])
                            cross_y = reference_values[i] + t * (reference_values[i+1] - reference_values[i])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
        
        if len(x_combined) > 0:
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
                line=dict(color=colors['reference'], width=line_width, dash='dash'),
                marker=dict(size=marker_size, color=colors['reference']),
                text=df['reference'].values.round(1),
                textposition='top center',
                textfont=dict(
                    family="Avenir",
                    size=annotation_size,
                    color="black"
                ),
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
    height=1200,
    showlegend=True,
    template='plotly_white',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    annotations=[
        dict(
            text=f'Glucose Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.09,
            showarrow=False
        ),
        dict(
            text=f'Insulin Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.48,
            showarrow=False
        )
    ],
    legend=dict(
        yanchor="top",
        y=1.18,
        xanchor="center",
        x=0.5,
        font=dict(family="Avenir"),
        orientation="h"
    ),
    font=dict(family="Avenir"),
    margin=dict(t=200, r=50, b=50, l=50)
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

# Display charts
st.markdown('<p class="subtitle">Visualization</p>', unsafe_allow_html=True)
with st.container():
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced download section
    col1, col2 = st.columns(2)
    with col1:
        if KALEIDO_INSTALLED:
            if st.button("Download Plot"):
                img_bytes = fig.to_image(
                    format=export_format, 
                    width=export_width, 
                    height=export_height, 
                    scale=export_scale
                )
                
                st.download_button(
                    label=f"Click to Download {export_format.upper()}",
                    data=img_bytes,
                    file_name=f"glucose_insulin_response.{export_format}",
                    mime=f"image/{export_format}"
                )
        else:
            st.info("Install kaleido for image export: pip install kaleido")
    
    with col2:
        # Export data as CSV
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="glucose_insulin_data.csv",
            mime="text/csv"
        )

# Enhanced Analysis Section
st.markdown('<p class="subtitle">Advanced Analysis</p>', unsafe_allow_html=True)

# Calculate advanced metrics
current_date = dates[0]
time_points = glucose_df['time'].values
current_glucose = glucose_df[current_date].values
current_insulin = insulin_df[current_date].values

# Area Under Curve calculations
glucose_auc = calculate_auc(time_points, current_glucose)
insulin_auc = calculate_auc(time_points, current_insulin)

# Metrics in enhanced containers
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    peak_glucose_change = max(current_glucose) - current_glucose[0]
    if len(dates) > 1:
        previous_glucose = glucose_df[dates[1]].values
        previous_peak_change = max(previous_glucose) - previous_glucose[0]
        delta = peak_glucose_change - previous_peak_change
    else:
        reference_glucose = glucose_df['reference'].values
        previous_peak_change = max(reference_glucose) - reference_glucose[0]
        delta = peak_glucose_change - previous_peak_change
    st.metric(
        "Peak Glucose Change",
        f"{peak_glucose_change:.1f} mg/dL",
        f"{delta:.1f} mg/dL vs previous"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    peak_insulin_change = max(current_insulin) - current_insulin[0]
    if len(dates) > 1:
        previous_insulin = insulin_df[dates[1]].values
        previous_peak_change = max(previous_insulin) - previous_insulin[0]
        delta = peak_insulin_change - previous_peak_change
    else:
        reference_insulin = insulin_df['reference'].values
        previous_peak_change = max(reference_insulin) - reference_insulin[0]
        delta = peak_insulin_change - previous_peak_change
    st.metric(
        "Peak Insulin Change",
        f"{peak_insulin_change:.1f} µU/mL",
        f"{delta:.1f} µU/mL vs previous"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Glucose AUC",
        f"{glucose_auc:.0f} mg·min/dL",
        help="Area under the glucose curve"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Insulin AUC",
        f"{insulin_auc:.0f} µU·min/mL",
        help="Area under the insulin curve"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Clinical Interpretation Section
if show_interpretation:
    st.markdown("### Clinical Interpretation")
    
    glucose_interpretations = interpret_glucose_response(current_glucose, time_points)
    insulin_interpretations = interpret_insulin_response(current_insulin, time_points)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="interpretation">', unsafe_allow_html=True)
        st.markdown("**Glucose Response:**")
        for interpretation in glucose_interpretations:
            st.markdown(f"• {interpretation}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="interpretation">', unsafe_allow_html=True)
        st.markdown("**Insulin Response:**")
        for interpretation in insulin_interpretations:
            st.markdown(f"• {interpretation}")
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Data Management
st.markdown('<p class="subtitle">Data Management</p>', unsafe_allow_html=True)

data_tab, column_tab, template_tab = st.tabs(["Row Operations", "Column Management", "Templates"])

with data_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Add New Test Date")
        new_month = st.selectbox("Month", 
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        new_year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
        new_date = f"{new_month} {new_year}"
        
        if st.button("Add New Date Column"):
            if new_date not in edited_df.columns:
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

with template_tab:
    st.markdown("#### Load Template Data")
    
    template_options = {
        "Standard OGTT": {
            'time': [0, 30, 60, 90, 120] * 2,
            'type': ['glucose'] * 5 + ['insulin'] * 5,
            'reference': [90, 140, 160, 140, 100] + [6, 40, 60, 40, 20]
        },
        "Extended OGTT": {
            'time': [0, 15, 30, 45, 60, 90, 120, 180] * 2,
            'type': ['glucose'] * 8 + ['insulin'] * 8,
            'reference': [90, 120, 140, 150, 160, 140, 120, 100] + [6, 25, 40, 50, 60, 40, 30, 20]
        }
    }
    
    selected_template = st.selectbox("Choose template", list(template_options.keys()))
    
    if st.button("Load Template"):
        template_data = template_options[selected_template].copy()
        template_data['Sample Date'] = [0] * len(template_data['time'])  # Placeholder values
        st.session_state.df = pd.DataFrame(template_data)
        st.success(f"Loaded {selected_template} template")
        st.rerun()

# Save changes with enhanced validation
if st.button("Save Changes"):
    valid = True
    error_msg = []
    
    # Enhanced validation
    for col in [c for c in edited_df.columns if c not in ['time', 'type']]:
        if col != 'reference':
            try:
                pd.to_numeric(edited_df[col], errors='raise')
            except ValueError:
                valid = False
                error_msg.append(f"Non-numeric values found in column {col}")
    
    if not all(t in ['glucose', 'insulin'] for t in edited_df['type']):
        valid = False
        error_msg.append("Invalid types found. Only 'glucose' and 'insulin' are allowed.")
    
    # Check for equal number of glucose and insulin rows
    glucose_count = (edited_df['type'] == 'glucose').sum()
    insulin_count = (edited_df['type'] == 'insulin').sum()
    if glucose_count != insulin_count:
        valid = False
        error_msg.append(f"Unequal number of glucose ({glucose_count}) and insulin ({insulin_count}) measurements")
    
    if valid:
        st.session_state.df = edited_df
        st.success("Changes saved successfully!")
        st.rerun()
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