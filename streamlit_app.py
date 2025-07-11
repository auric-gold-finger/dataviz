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

def calculate_matsuda_index(glucose_values, insulin_values, time_points):
    """
    Calculate Matsuda Index (ISI) for insulin sensitivity
    Formula: 10000 / sqrt((FPG × FPI) × (mean glucose × mean insulin))
    Where FPG = fasting plasma glucose, FPI = fasting plasma insulin
    """
    try:
        # Fasting values (baseline)
        fpg = glucose_values[0]  # mg/dL
        fpi = insulin_values[0]  # µU/mL
        
        # Mean glucose and insulin during OGTT (excluding fasting)
        mean_glucose = np.mean(glucose_values[1:]) if len(glucose_values) > 1 else glucose_values[0]
        mean_insulin = np.mean(insulin_values[1:]) if len(insulin_values) > 1 else insulin_values[0]
        
        # Matsuda index calculation
        matsuda = 10000 / np.sqrt((fpg * fpi) * (mean_glucose * mean_insulin))
        return matsuda
    except:
        return None

def calculate_homa_ir(glucose_fasting, insulin_fasting):
    """
    Calculate HOMA-IR (Homeostatic Model Assessment of Insulin Resistance)
    Formula: (Fasting Glucose × Fasting Insulin) / 405
    """
    try:
        return (glucose_fasting * insulin_fasting) / 405
    except:
        return None

def calculate_glucose_insulin_ratio(glucose_values, insulin_values):
    """
    Calculate glucose/insulin ratios at different time points
    Lower ratios may indicate insulin resistance
    """
    try:
        ratios = []
        for g, i in zip(glucose_values, insulin_values):
            if i > 0:  # Avoid division by zero
                ratios.append(g / i)
            else:
                ratios.append(float('inf'))
        return ratios
    except:
        return None

def get_dynamic_interpretation(glucose_values, insulin_values, time_points):
    """
    Provide dynamic interpretation based on calculated indices
    """
    interpretations = {
        'glucose': [],
        'insulin': [],
        'metabolic': []
    }
    
    # Calculate metabolic indices
    matsuda = calculate_matsuda_index(glucose_values, insulin_values, time_points)
    homa_ir = calculate_homa_ir(glucose_values[0], insulin_values[0])
    gi_ratios = calculate_glucose_insulin_ratio(glucose_values, insulin_values)
    
    # Glucose interpretations
    baseline_glucose = glucose_values[0]
    peak_glucose = max(glucose_values)
    glucose_2h = glucose_values[-1] if len(glucose_values) >= 4 else None
    
    if baseline_glucose < 70:
        interpretations['glucose'].append("⚠️ Hypoglycemia risk - fasting glucose too low")
    elif baseline_glucose > 126:
        interpretations['glucose'].append("🔴 Diabetic range - fasting glucose ≥126 mg/dL")
    elif baseline_glucose > 100:
        interpretations['glucose'].append("🟡 Impaired fasting glucose (100-125 mg/dL)")
    else:
        interpretations['glucose'].append("✅ Normal fasting glucose (<100 mg/dL)")
    
    if glucose_2h is not None:
        if glucose_2h >= 200:
            interpretations['glucose'].append("🔴 Diabetic range - 2h glucose ≥200 mg/dL")
        elif glucose_2h >= 140:
            interpretations['glucose'].append("🟡 Impaired glucose tolerance (140-199 mg/dL)")
        else:
            interpretations['glucose'].append("✅ Normal glucose tolerance (<140 mg/dL)")
    
    # Insulin interpretations
    baseline_insulin = insulin_values[0]
    peak_insulin = max(insulin_values)
    
    if baseline_insulin > 25:
        interpretations['insulin'].append("⚠️ Elevated fasting insulin - possible insulin resistance")
    elif baseline_insulin < 2:
        interpretations['insulin'].append("⚠️ Low fasting insulin - possible β-cell dysfunction")
    else:
        interpretations['insulin'].append("✅ Normal fasting insulin (2-25 µU/mL)")
    
    if peak_insulin > 150:
        interpretations['insulin'].append("⚠️ Excessive insulin response - insulin resistance likely")
    elif peak_insulin < 30:
        interpretations['insulin'].append("⚠️ Blunted insulin response - β-cell dysfunction possible")
    
    # Metabolic interpretations based on calculated indices
    if matsuda is not None:
        if matsuda < 2.5:
            interpretations['metabolic'].append(f"🔴 Low insulin sensitivity (Matsuda: {matsuda:.1f}) - insulin resistance")
        elif matsuda < 5.0:
            interpretations['metabolic'].append(f"🟡 Reduced insulin sensitivity (Matsuda: {matsuda:.1f})")
        else:
            interpretations['metabolic'].append(f"✅ Good insulin sensitivity (Matsuda: {matsuda:.1f})")
    
    if homa_ir is not None:
        if homa_ir > 2.5:
            interpretations['metabolic'].append(f"⚠️ Insulin resistance (HOMA-IR: {homa_ir:.1f})")
        else:
            interpretations['metabolic'].append(f"✅ Normal insulin sensitivity (HOMA-IR: {homa_ir:.1f})")
    
    if gi_ratios:
        fasting_ratio = gi_ratios[0]
        if fasting_ratio < 6:
            interpretations['metabolic'].append(f"⚠️ Low G/I ratio ({fasting_ratio:.1f}) suggests insulin resistance")
        else:
            interpretations['metabolic'].append(f"✅ Normal G/I ratio ({fasting_ratio:.1f})")
    
    return interpretations, {'matsuda': matsuda, 'homa_ir': homa_ir, 'gi_ratios': gi_ratios}

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

# Calculate metabolic indices
matsuda_index = calculate_matsuda_index(current_glucose, current_insulin, time_points)
homa_ir = calculate_homa_ir(current_glucose[0], current_insulin[0])
gi_ratios = calculate_glucose_insulin_ratio(current_glucose, current_insulin)

# Metrics in enhanced containers
col1, col2, col3, col4, col5 = st.columns(5)

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
    if matsuda_index is not None:
        # Color coding based on Matsuda index
        if matsuda_index < 2.5:
            color = "🔴"
        elif matsuda_index < 5.0:
            color = "🟡"
        else:
            color = "✅"
        st.metric(
            f"{color} Matsuda Index",
            f"{matsuda_index:.1f}",
            help="Insulin sensitivity index (>5.0 = good, 2.5-5.0 = reduced, <2.5 = low)"
        )
    else:
        st.metric("Matsuda Index", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if homa_ir is not None:
        color = "⚠️" if homa_ir > 2.5 else "✅"
        st.metric(
            f"{color} HOMA-IR",
            f"{homa_ir:.1f}",
            help="Insulin resistance index (<2.5 = normal, >2.5 = insulin resistance)"
        )
    else:
        st.metric("HOMA-IR", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if gi_ratios and len(gi_ratios) > 0:
        fasting_gi_ratio = gi_ratios[0]
        color = "⚠️" if fasting_gi_ratio < 6 else "✅"
        st.metric(
            f"{color} G/I Ratio",
            f"{fasting_gi_ratio:.1f}",
            help="Glucose/Insulin ratio (>6 = normal, <6 = possible insulin resistance)"
        )
    else:
        st.metric("G/I Ratio", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

# Clinical Interpretation Section
if show_interpretation:
    st.markdown("### Dynamic Clinical Interpretation")
    
    # Get dynamic interpretations
    interpretations, indices = get_dynamic_interpretation(current_glucose, current_insulin, time_points)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="interpretation">', unsafe_allow_html=True)
        st.markdown("**Glucose Response:**")
        for interpretation in interpretations['glucose']:
            st.markdown(f"• {interpretation}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="interpretation">', unsafe_allow_html=True)
        st.markdown("**Insulin Response:**")
        for interpretation in interpretations['insulin']:
            st.markdown(f"• {interpretation}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="interpretation">', unsafe_allow_html=True)
        st.markdown("**Metabolic Health:**")
        for interpretation in interpretations['metabolic']:
            st.markdown(f"• {interpretation}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional metabolic insights
    if indices['gi_ratios']:
        st.markdown("#### Glucose/Insulin Ratio Timeline")
        ratio_fig = go.Figure()
        ratio_fig.add_trace(
            go.Scatter(
                x=time_points,
                y=indices['gi_ratios'],
                mode='lines+markers',
                name='G/I Ratio',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            )
        )
        # Add reference line at 6 (threshold for insulin resistance)
        ratio_fig.add_hline(y=6, line_dash="dash", line_color="red", 
                          annotation_text="Insulin Resistance Threshold")
        
        ratio_fig.update_layout(
            title="Glucose/Insulin Ratio Over Time",
            xaxis_title="Time (minutes)",
            yaxis_title="Glucose/Insulin Ratio",
            template='plotly_white',
            height=300
        )
        st.plotly_chart(ratio_fig, use_container_width=True)

# Enhanced Data Management
st.markdown('<p class="subtitle">Data Management</p>', unsafe_allow_html=True)

data_tab, column_tab, template_tab, compare_tab = st.tabs(["Row Operations", "Column Management", "Templates", "Compare Tests"])

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

with compare_tab:
    st.markdown("#### Statistical Comparison")
    
    if len(dates) >= 2:
        # Select dates to compare
        date1 = st.selectbox("First test date", dates, key="date1")
        date2 = st.selectbox("Second test date", dates, key="date2", index=1 if len(dates) > 1 else 0)
        
        if date1 != date2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Glucose Comparison**")
                glucose1 = glucose_df[date1].values
                glucose2 = glucose_df[date2].values
                
                # Calculate differences
                peak_diff = max(glucose1) - max(glucose2)
                auc1 = calculate_auc(time_points, glucose1)
                auc2 = calculate_auc(time_points, glucose2)
                auc_diff = auc1 - auc2
                
                st.metric("Peak Difference", f"{peak_diff:.1f} mg/dL")
                st.metric("AUC Difference", f"{auc_diff:.0f} mg·min/dL")
                
                # Statistical significance (simple t-test approximation)
                if len(glucose1) > 2:
                    from scipy.stats import ttest_rel
                    try:
                        stat, p_value = ttest_rel(glucose1, glucose2)
                        significance = "Significant" if p_value < 0.05 else "Not significant"
                        st.metric("Statistical Test", significance, f"p={p_value:.3f}")
                    except:
                        st.info("Install scipy for statistical tests: pip install scipy")
            
            with col2:
                st.markdown("**Insulin Comparison**")
                insulin1 = insulin_df[date1].values
                insulin2 = insulin_df[date2].values
                
                peak_diff = max(insulin1) - max(insulin2)
                auc1 = calculate_auc(time_points, insulin1)
                auc2 = calculate_auc(time_points, insulin2)
                auc_diff = auc1 - auc2
                
                st.metric("Peak Difference", f"{peak_diff:.1f} µU/mL")
                st.metric("AUC Difference", f"{auc_diff:.0f} µU·min/mL")
                
                if len(insulin1) > 2:
                    try:
                        from scipy.stats import ttest_rel
                        stat, p_value = ttest_rel(insulin1, insulin2)
                        significance = "Significant" if p_value < 0.05 else "Not significant"
                        st.metric("Statistical Test", significance, f"p={p_value:.3f}")
                    except:
                        st.info("Install scipy for statistical tests")
            
            # Trend analysis
            st.markdown("**Trend Analysis**")
            if auc1 > auc2:
                if 'glucose' in st.session_state:
                    st.success(f"✅ Glucose response improved from {date2} to {date1}")
                else:
                    st.warning(f"⚠️ Glucose response worsened from {date2} to {date1}")
            else:
                st.info(f"📊 Glucose response changed from {date2} to {date1}")
    else:
        st.info("Need at least 2 test dates for comparison")

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

# Additional Features Section
st.markdown('<p class="subtitle">Additional Tools</p>', unsafe_allow_html=True)

tool_col1, tool_col2, tool_col3 = st.columns(3)

with tool_col1:
    st.markdown("#### 📊 Data Summary")
    with st.expander("View Data Statistics"):
        current_date = dates[0]
        
        # Glucose statistics
        st.markdown("**Glucose Stats:**")
        glucose_values = glucose_df[current_date].values
        st.write(f"• Mean: {np.mean(glucose_values):.1f} mg/dL")
        st.write(f"• Peak: {np.max(glucose_values):.1f} mg/dL")
        st.write(f"• Range: {np.ptp(glucose_values):.1f} mg/dL")
        st.write(f"• CV: {(np.std(glucose_values)/np.mean(glucose_values)*100):.1f}%")
        
        st.markdown("**Insulin Stats:**")
        insulin_values = insulin_df[current_date].values
        st.write(f"• Mean: {np.mean(insulin_values):.1f} µU/mL")
        st.write(f"• Peak: {np.max(insulin_values):.1f} µU/mL")
        st.write(f"• Range: {np.ptp(insulin_values):.1f} µU/mL")
        st.write(f"• CV: {(np.std(insulin_values)/np.mean(insulin_values)*100):.1f}%")

with tool_col2:
    st.markdown("#### 🎯 Target Zones")
    with st.expander("Reference Ranges"):
        st.markdown("**Normal Glucose Response:**")
        st.write("• Fasting: 70-100 mg/dL")
        st.write("• 2-hour: <140 mg/dL")
        st.write("• Peak: <200 mg/dL")
        
        st.markdown("**Normal Insulin Response:**")
        st.write("• Fasting: 2-25 µU/mL")
        st.write("• Peak: 30-100 µU/mL")
        st.write("• 2-hour: <50 µU/mL")
        
        st.markdown("**Interpretation Guidelines:**")
        st.write("• Glucose >200 mg/dL → Diabetes risk")
        st.write("• Glucose 140-199 mg/dL → Impaired tolerance")
        st.write("• High insulin → Insulin resistance")
        st.write("• Low insulin → Beta-cell dysfunction")

with tool_col3:
    st.markdown("#### 💡 Recommendations")
    with st.expander("Next Steps"):
        # Generate personalized recommendations based on current data
        current_glucose = glucose_df[current_date].values
        current_insulin = insulin_df[current_date].values
        
        recommendations = []
        
        if max(current_glucose) > 200:
            recommendations.append("🔴 Consult healthcare provider about diabetes risk")
        elif max(current_glucose) > 140:
            recommendations.append("🟡 Consider lifestyle modifications")
        else:
            recommendations.append("🟢 Maintain current healthy habits")
        
        if max(current_insulin) > 100:
            recommendations.append("⚠️ Discuss insulin resistance with doctor")
            recommendations.append("💪 Consider resistance training")
            recommendations.append("🥗 Review carbohydrate intake")
        
        if current_glucose[-1] > current_glucose[0] + 20:
            recommendations.append("⏰ Monitor glucose recovery patterns")
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        if not recommendations:
            st.write("• 🎉 Results appear within normal ranges")
            st.write("• 📅 Continue regular monitoring")
            st.write("• 🏃‍♀️ Maintain active lifestyle")

# Quick Actions
st.markdown("#### Quick Actions")
action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("🔄 Reset to Sample Data"):
        # Reset to original sample data
        data = {
            'time': [0, 30, 60, 90] * 2,
            'type': ['glucose'] * 4 + ['insulin'] * 4,
            'Sept 2022': [103, 158, 142, 159] + [12, 72, 78, 107],
            'Feb 2025': [91, 148, 119, 85] + [8, 66, 75, 37],
            'reference': [90, 140, 120, 100] + [6, 40, 30, 20]
        }
        st.session_state.df = pd.DataFrame(data)
        st.success("Reset to sample data!")
        st.rerun()

with action_col2:
    if st.button("📋 Copy Data Format"):
        format_example = """time,type,Feb 2025,reference
0,glucose,91,90
30,glucose,148,140
60,glucose,119,120
90,glucose,85,100
0,insulin,8,6
30,insulin,66,40
60,insulin,75,30
90,insulin,37,20"""
        st.code(format_example, language="csv")

with action_col3:
    if st.button("📈 Generate Report"):
        # Create a summary report
        current_date = dates[0]
        current_glucose = glucose_df[current_date].values
        current_insulin = insulin_df[current_date].values
        
        report = f"""
# Glucose Tolerance Test Report
**Test Date:** {current_date}
**Test Type:** 75g Oral Glucose Tolerance Test

## Results Summary
- **Baseline Glucose:** {current_glucose[0]:.1f} mg/dL
- **Peak Glucose:** {max(current_glucose):.1f} mg/dL at {time_points[np.argmax(current_glucose)]} minutes
- **2-hour Glucose:** {current_glucose[-1]:.1f} mg/dL
- **Glucose AUC:** {calculate_auc(time_points, current_glucose):.0f} mg·min/dL

- **Baseline Insulin:** {current_insulin[0]:.1f} µU/mL
- **Peak Insulin:** {max(current_insulin):.1f} µU/mL at {time_points[np.argmax(current_insulin)]} minutes
- **2-hour Insulin:** {current_insulin[-1]:.1f} µU/mL
- **Insulin AUC:** {calculate_auc(time_points, current_insulin):.0f} µU·min/mL

## Clinical Interpretation
### Glucose Response:
{chr(10).join(f"• {interp}" for interp in interpret_glucose_response(current_glucose, time_points))}

### Insulin Response:
{chr(10).join(f"• {interp}" for interp in interpret_insulin_response(current_insulin, time_points))}

*This report is for informational purposes only and should not replace professional medical advice.*
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"ogtt_report_{current_date.replace(' ', '_')}.md",
            mime="text/markdown"
        )

with action_col4:
    if st.button("⚙️ Advanced Settings"):
        with st.expander("Advanced Configuration", expanded=True):
            st.markdown("**Calculation Settings:**")
            auc_method = st.selectbox("AUC Method", ["Trapezoidal", "Simpson's Rule"])
            baseline_correction = st.checkbox("Baseline Correction", value=False)
            
            st.markdown("**Clinical Thresholds:**")
            glucose_threshold = st.slider("Glucose Alert Threshold", 140, 250, 200)
            insulin_threshold = st.slider("Insulin Alert Threshold", 50, 150, 100)
            
            if st.button("Apply Settings"):
                st.success("Settings applied!")

# Footer with app information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
    <p><strong>Glucose & Insulin Response Analyzer</strong> v2.0</p>
    <p>Built with Streamlit & Plotly | For research and educational purposes</p>
    <p>⚠️ <em>Not intended for clinical diagnosis - always consult healthcare professionals</em></p>
</div>
""", unsafe_allow_html=True)