import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="OGTT Analyzer",
    page_icon="🩺",
    layout="wide"
)

# Simplified CSS with typography focus
st.markdown("""
<style>
/* Typography imports */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Avenir:wght@400;500&display=swap');

/* Clean, minimal styling */
.main > div {
    padding-top: 1rem;
}

/* Compact input styling */
div[data-testid="stNumberInput"] {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 4px;
    margin-bottom: 0.5rem;
}

/* Metric cards - clean and simple */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #f0f0f0;
    border-radius: 8px;
    padding: 16px;
}

/* Simple header */
.simple-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
    font-family: 'Cormorant Garamond', serif;
}

.simple-header h1 {
    font-size: 2rem;
    color: #2c3e50;
    margin-bottom: 0.25rem;
}

.simple-header p {
    font-family: 'Avenir', sans-serif;
    color: #7f8c8d;
    font-size: 0.9rem;
}

/* Result boxes */
.result-box {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid;
    background: white;
}

.normal { border-left-color: #27ae60; }
.warning { border-left-color: #f39c12; }
.concern { border-left-color: #e74c3c; }

/* Test sections */
.test-section {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for multiple tests
if 'test_data' not in st.session_state:
    st.session_state.test_data = {}

# Compact header
st.markdown("""
<div class="simple-header">
    <h1>OGTT Analyzer</h1>
    <p>Multi-Test OGTT Analysis with Simultaneous Graphing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Test management
    st.subheader("Test Management")
    
    # CSV Upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV Preview:")
            st.dataframe(df.head())
            
            if st.button("Load from CSV"):
                st.session_state.test_data = {}
                
                # Expected columns: Date, Time_0, Time_30, Time_60, Time_90, Glucose_0, Glucose_30, Glucose_60, Glucose_90, Insulin_0, Insulin_30, Insulin_60, Insulin_90
                colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A2B8', '#FD7E14', '#6C757D', '#DC3545', '#20C997']
                
                for i, row in df.iterrows():
                    test_id = i + 1
                    color = colors[i % len(colors)]
                    
                    # Parse times - use defaults if not provided
                    if 'Time_0' in df.columns:
                        times = [row.get('Time_0', 0), row.get('Time_30', 30), row.get('Time_60', 60), row.get('Time_90', 90)]
                    else:
                        times = [0, 30, 60, 90]
                    
                    # Parse glucose values
                    glucose = [
                        row.get('Glucose_0', 88),
                        row.get('Glucose_30', 119),
                        row.get('Glucose_60', 92),
                        row.get('Glucose_90', 80)
                    ]
                    
                    # Parse insulin values
                    insulin = [
                        row.get('Insulin_0', 8),
                        row.get('Insulin_30', 50),
                        row.get('Insulin_60', 70),
                        row.get('Insulin_90', 40)
                    ]
                    
                    # Parse date
                    date_str = row.get('Date', datetime.now().strftime('%Y-%m-%d'))
                    try:
                        parsed_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
                    except:
                        parsed_date = datetime.now().strftime('%Y-%m-%d')
                    
                    st.session_state.test_data[test_id] = {
                        'date': parsed_date,
                        'times': times,
                        'glucose': glucose,
                        'insulin': insulin,
                        'color': color
                    }
                
                st.success(f"Loaded {len(df)} tests from CSV")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            st.info("Expected CSV format: Date, Glucose_0, Glucose_30, Glucose_60, Glucose_90, Insulin_0, Insulin_30, Insulin_60, Insulin_90")
    
    # Manual test input
    num_tests = st.number_input("Number of tests:", min_value=1, max_value=10, value=1, step=1)
    
    if st.button("Clear All Tests"):
        st.session_state.test_data = {}
        st.rerun()
    
    # Chart appearance
    with st.expander("Chart Appearance", expanded=True):
        line_width = st.slider("Line width", 1, 10, 4)
        marker_size = st.slider("Marker size", 4, 20, 10)
        show_shading = st.checkbox("Show comparative shading", value=True)
        show_reference = st.checkbox("Show reference line", value=True)
        show_legend = st.checkbox("Show legend", value=True)
    
    # Test visibility toggles
    if st.session_state.test_data:
        st.subheader("Test Visibility")
        test_visibility = {}
        for test_id in st.session_state.test_data.keys():
            test_date = st.session_state.test_data[test_id].get('date', f'Test {test_id}')
            test_visibility[test_id] = st.checkbox(f"Show {test_date}", value=True, key=f"vis_{test_id}")
    else:
        test_visibility = {}
    
    # Reference line details
    with st.expander("Reference Line Details"):
        reference_color = st.color_picker("Reference line color", "#1cb164")
        reference_opacity = st.slider("Reference line opacity", 0.1, 1.0, 0.5, 0.1)
        reference_width = st.slider("Reference line width", 1, 10, 3)
    
    # Annotations
    with st.expander("Annotations"):
        show_annotations = st.checkbox("Show value annotations", value=True)
        annotation_size = st.slider("Text size", 8, 48, 12)
        annotation_bold = st.checkbox("Bold text", value=True)
        annotation_color = st.color_picker("Text color", "#2c3e50")
        annotation_bg_color = st.color_picker("Background color", "#ffffff")
        annotation_border_color = st.color_picker("Border color", "#cccccc")
    
    # Typography
    with st.expander("Typography"):
        title_font_size = st.slider("Chart title size", 12, 72, 32)
        axis_title_size = st.slider("Axis title size", 8, 48, 20)
        axis_tick_size = st.slider("Axis tick size", 6, 36, 16)
    
    # Export settings
    with st.expander("Export Settings"):
        download_width = st.slider("Width (px)", 800, 4000, 1200, step=200)
        download_height = st.slider("Height (px)", 400, 3000, 550, step=50)
        download_scale = st.slider("Scale multiplier", 1, 10, 1)

# Test input section
st.markdown("### Test Data Entry")

# Color palette for tests
colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A2B8', '#FD7E14', '#6C757D', '#DC3545', '#20C997']

# Create input sections for each test
for i in range(num_tests):
    test_id = i + 1
    color = colors[i % len(colors)]
    
    with st.expander(f"Test {test_id}", expanded=True):
        # Only date input - no separate name field
        test_date = st.date_input(
            "Date:",
            value=datetime.strptime(st.session_state.test_data.get(test_id, {}).get('date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d').date(),
            key=f"date_{test_id}"
        )
        
        # Data input
        col1, col2, col3 = st.columns(3)
        
        default_times = st.session_state.test_data.get(test_id, {}).get('times', [0, 30, 60, 90])
        default_glucose = st.session_state.test_data.get(test_id, {}).get('glucose', [88, 119, 92, 80])
        default_insulin = st.session_state.test_data.get(test_id, {}).get('insulin', [8, 50, 70, 40])
        
        with col1:
            st.markdown("**Time Points (min)**")
            time_input = st.text_input(
                "", 
                value=", ".join(map(str, default_times)),
                key=f"times_{test_id}"
            )
            
        with col2:
            st.markdown("**Glucose (mg/dL)**")
            glucose_input = st.text_input(
                "", 
                value=", ".join(map(str, default_glucose)),
                key=f"glucose_{test_id}"
            )
            
        with col3:
            st.markdown("**Insulin (μU/mL)**")
            insulin_input = st.text_input(
                "", 
                value=", ".join(map(str, default_insulin)),
                key=f"insulin_{test_id}"
            )
        
        # Parse and auto-save data
        try:
            times = [float(x.strip()) for x in time_input.split(',')]
            glucose_vals = [float(x.strip()) for x in glucose_input.split(',')]
            insulin_vals = [float(x.strip()) for x in insulin_input.split(',')]
            
            if len(times) >= 4 and len(glucose_vals) >= 4 and len(insulin_vals) >= 4:
                # Auto-save to session state
                st.session_state.test_data[test_id] = {
                    'date': test_date.strftime('%Y-%m-%d'),
                    'times': times[:4],
                    'glucose': glucose_vals[:4],
                    'insulin': insulin_vals[:4],
                    'color': color
                }
            else:
                if test_id not in st.session_state.test_data:
                    st.session_state.test_data[test_id] = {
                        'date': test_date.strftime('%Y-%m-%d'),
                        'times': [0, 30, 60, 90],
                        'glucose': [88, 119, 92, 80],
                        'insulin': [8, 50, 70, 40],
                        'color': color
                    }
        except:
            if test_id not in st.session_state.test_data:
                st.session_state.test_data[test_id] = {
                    'date': test_date.strftime('%Y-%m-%d'),
                    'times': [0, 30, 60, 90],
                    'glucose': [88, 119, 92, 80],
                    'insulin': [8, 50, 70, 40],
                    'color': color
                }

# Reference patterns
normal_glucose = [90, 140, 120, 95]
normal_insulin = [6, 40, 30, 20]

# IMPROVED CLASSIFICATION ALGORITHMS
def calculate_glucotype(glucose_vals):
    """
    Improved glucotype classification based on Hall et al. research
    """
    g0, g30, g60, g90 = glucose_vals
    
    # Calculate incremental changes
    delta_30 = g30 - g0
    delta_60 = g60 - g30
    delta_90 = g90 - g60
    
    # Additional metrics
    peak_glucose = max(glucose_vals)
    peak_time = glucose_vals.index(peak_glucose) * 30
    returns_to_baseline = g90 <= (g0 + 10)
    
    # Classify based on response patterns
    if delta_30 <= 0:
        return "Flat/Low Responder", "concern"
    elif peak_glucose < 100:
        return "Flat/Low Responder", "concern"
    elif delta_30 > 0 and delta_60 > 0 and delta_90 > 0:
        return "Continuous Rise", "concern"
    elif peak_time == 30:
        if delta_60 <= -20 and delta_90 > 10:
            return "Biphasic", "normal"
        elif delta_60 <= 0 and returns_to_baseline:
            return "Monophasic", "warning"
        elif delta_60 > 0:
            return "Continuous Rise", "concern"
        else:
            return "Monophasic", "warning"
    elif peak_time == 60:
        if delta_90 <= -15:
            if returns_to_baseline:
                return "Delayed Monophasic", "warning"
            else:
                return "Delayed Peak", "warning"
        else:
            return "Sustained Elevation", "concern"
    elif peak_time == 90:
        return "Late Peak/Continuous Rise", "concern"
    else:
        if g30 > g0 and g60 < g30 and g90 > g60 and g90 < g30:
            return "Biphasic", "normal"
        else:
            return "Atypical Pattern", "warning"

def calculate_kraft_type(insulin_vals, glucose_vals=None):
    """
    Enhanced Kraft classification with glucose context
    Key principle: Low insulin + good glucose control = high insulin sensitivity
    """
    i0, i30, i60, i90 = insulin_vals
    
    # Calculate glucose metrics if available
    glucose_well_controlled = False
    glucose_disposal_excellent = False
    if glucose_vals:
        g0, g30, g60, g90 = glucose_vals
        glucose_peak = max(glucose_vals)
        glucose_well_controlled = glucose_peak < 140 and g90 <= g0 + 15
        glucose_disposal_excellent = glucose_peak < 120 and g90 <= g0 + 10
    
    # Type IV: High fasting insulin
    if i0 >= 25:
        return "Type IV - High Fasting", "concern"
    
    # Enhanced Type V analysis - consider glucose control
    peak_insulin = max(insulin_vals)
    if peak_insulin < 30:
        if glucose_disposal_excellent:
            return "Type I - Exceptional Sensitivity", "normal"
        elif glucose_well_controlled:
            return "Type I - High Sensitivity", "normal"
        else:
            return "Type V - Low Response", "concern"
    
    # Find peak time and characteristics
    peak_time_index = insulin_vals.index(peak_insulin)
    peak_time = peak_time_index * 30
    
    # Enhanced Type I criteria
    if peak_time == 30:
        if i90 < (i0 + 15):
            if i90 < 30 and glucose_disposal_excellent:
                return "Type I - Exceptional", "normal"
            elif i90 < 40:
                return "Type I - Normal", "normal"
            elif i90 < 60:
                return "Type I/II - Borderline", "warning"
            else:
                return "Type II - Early Pattern", "warning"
        elif i90 >= 60:
            return "Type III - Sustained", "warning"
        else:
            return "Type II - Early Pattern", "warning"
    
    # Type II: Delayed peak
    elif peak_time >= 60:
        if i90 < 50 and glucose_well_controlled:
            return "Type II - Delayed Normal", "warning"
        elif i90 >= 60:
            return "Type II/III - Delayed Sustained", "concern"
        else:
            return "Type II - Delayed Peak", "warning"
    
    # Type III: Sustained elevation
    elif i90 >= 60 or i90 > (i0 + 30):
        return "Type III - Sustained", "warning"
    
    # Default classification
    else:
        if i90 < 40 and glucose_well_controlled:
            return "Type I - Normal", "normal"
        elif i90 < 50:
            return "Type I/II - Borderline", "warning"
        else:
            return "Type II - Early Pattern", "warning"

def calculate_metrics(glucose_vals, insulin_vals):
    """Enhanced metabolic metrics calculation"""
    g0, g30, g60, g90 = glucose_vals
    i0, i30, i60, i90 = insulin_vals
    
    # HOMA-IR
    homa_ir = (g0 * i0) / 405
    
    # Matsuda Index
    mean_glucose = np.mean(glucose_vals)
    mean_insulin = np.mean(insulin_vals)
    matsuda = 10000 / math.sqrt(g0 * i0 * mean_glucose * mean_insulin)
    
    # Area under curve
    glucose_auc = 0.5 * (g0 + g30) * 30 + 0.5 * (g30 + g60) * 30 + 0.5 * (g60 + g90) * 30
    insulin_auc = 0.5 * (i0 + i30) * 30 + 0.5 * (i30 + i60) * 30 + 0.5 * (i60 + i90) * 30
    
    # Additional metrics
    glucose_peak = max(glucose_vals)
    insulin_peak = max(insulin_vals)
    
    # Disposition index
    insulinogenic_index = (i30 - i0) / (g30 - g0) if g30 > g0 else 0
    disposition_index = insulinogenic_index * matsuda
    
    return {
        'homa_ir': homa_ir,
        'matsuda': matsuda,
        'glucose_auc': glucose_auc,
        'insulin_auc': insulin_auc,
        'glucose_peak': glucose_peak,
        'insulin_peak': insulin_peak,
        'insulinogenic_index': insulinogenic_index,
        'disposition_index': disposition_index
    }

# Multi-test plotting function
def create_multi_test_plot(title, y_label, reference_data, reference_name, is_glucose=True):
    fig = go.Figure()
    
    # Add reference line first
    if show_reference:
        fig.add_trace(go.Scatter(
            x=[0, 30, 60, 90], 
            y=reference_data,
            mode='lines',
            name=f'{reference_name} Pattern',
            line=dict(color=reference_color, width=reference_width, dash='dot'),
            opacity=reference_opacity
        ))
    
    # Add each test
    for test_id, test_data in st.session_state.test_data.items():
        if test_id in test_visibility and test_visibility[test_id]:
            test_times = test_data['times']
            test_values = test_data['glucose'] if is_glucose else test_data['insulin']
            test_color = test_data['color']
            test_date = test_data['date']
            
            # Add shading for this test if enabled
            if show_shading and is_glucose:  # Only shade glucose for clarity
                for i in range(len(test_times) - 1):
                    x1, x2 = test_times[i], test_times[i+1]
                    y1_user, y2_user = test_values[i], test_values[i+1]
                    y1_ref, y2_ref = reference_data[i], reference_data[i+1]
                    
                    user_above = (y1_user + y2_user) / 2 > (y1_ref + y2_ref) / 2
                    
                    if user_above:
                        fill_color = f'rgba(231, 76, 60, 0.05)'
                    else:
                        fill_color = f'rgba(46, 204, 113, 0.03)'
                    
                    x_fill = [x1, x2, x2, x1]
                    if user_above:
                        y_fill = [y1_ref, y2_ref, y2_user, y1_user]
                    else:
                        y_fill = [y1_user, y2_user, y2_ref, y1_ref]
                    
                    fig.add_trace(go.Scatter(
                        x=x_fill, y=y_fill, fill='toself', fillcolor=fill_color,
                        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
                    ))
            
            # Add test line
            fig.add_trace(go.Scatter(
                x=test_times, 
                y=test_values,
                mode='lines+markers',
                name=test_date,  # Use date as legend name
                line=dict(color=test_color, width=line_width),
                marker=dict(size=marker_size, color=test_color, line=dict(color='white', width=2))
            ))
            
            # Add annotations for this test with boxes
            if show_annotations:
                for i, (x, y) in enumerate(zip(test_times, test_values)):
                    text_content = f"<b>{y:.0f}</b>" if annotation_bold else f"{y:.0f}"
                    
                    fig.add_annotation(
                        x=x, y=y, text=text_content, 
                        showarrow=False, 
                        yshift=25,
                        font=dict(family="Avenir, sans-serif", size=annotation_size, color=annotation_color),
                        bgcolor=annotation_bg_color,
                        bordercolor=annotation_border_color,
                        borderwidth=1,
                        borderpad=4
                    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Cormorant Garamond, serif", size=title_font_size, color='#2c3e50'),
            x=0.5, xanchor='center'
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title=dict(text="Time (minutes)", font=dict(family="Cormorant Garamond, serif", size=axis_title_size, color='#34495e')),
            tickfont=dict(family="Avenir, sans-serif", size=axis_tick_size, color='#7f8c8d'),
            showgrid=True, gridcolor='#ecf0f1', gridwidth=1,
            tickmode='array', tickvals=[0, 30, 60, 90], ticktext=['0', '30', '60', '90']
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(family="Cormorant Garamond, serif", size=axis_title_size, color='#34495e')),
            tickfont=dict(family="Avenir, sans-serif", size=axis_tick_size, color='#7f8c8d'),
            showgrid=True, gridcolor='#ecf0f1', gridwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=40, l=60, r=40), font=dict(family="Avenir, sans-serif")
    )
    
    return fig

# Create plots
if st.session_state.test_data:
    st.markdown("### Glucose Response - All Tests")
    fig1 = create_multi_test_plot(
        'Glucose Response Comparison', 'Glucose (mg/dL)',
        normal_glucose, 'Normal', is_glucose=True
    )
    
    st.plotly_chart(fig1, use_container_width=True, config={
        'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage'],
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'glucose_comparison',
            'height': download_height, 'width': download_width, 'scale': download_scale
        }
    })
    
    st.markdown("### Insulin Response - All Tests")
    fig2 = create_multi_test_plot(
        'Insulin Response Comparison', 'Insulin (μU/mL)',
        normal_insulin, 'Normal', is_glucose=False
    )
    
    st.plotly_chart(fig2, use_container_width=True, config={
        'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage'],
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'insulin_comparison',
            'height': download_height, 'width': download_width, 'scale': download_scale
        }
    })
    
    # Analysis for each test
    st.markdown("### Individual Test Analysis")
    
    for test_id, test_data in st.session_state.test_data.items():
        if test_id in test_visibility and test_visibility[test_id]:
            glucose = test_data['glucose']
            insulin = test_data['insulin']
            test_date = test_data['date']
            
            # Calculate results
            glucotype, gluco_status = calculate_glucotype(glucose)
            kraft_type, kraft_status = calculate_kraft_type(insulin, glucose)
            metrics = calculate_metrics(glucose, insulin)
            
            # Determine HOMA-IR status
            if metrics['homa_ir'] < 1.0:
                homa_status = "normal"
            elif metrics['homa_ir'] < 2.5:
                homa_status = "warning"
            else:
                homa_status = "concern"
            
            # Display results for this test
            st.markdown(f"#### {test_date}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="result-box {gluco_status}">
                    <strong>Glucotype</strong><br>
                    {glucotype}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-box {kraft_status}">
                    <strong>Kraft Pattern</strong><br>
                    {kraft_type}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="result-box {homa_status}">
                    <strong>HOMA-IR</strong><br>
                    {metrics['homa_ir']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Matsuda Index", f"{metrics['matsuda']:.1f}")
                st.metric("Peak Glucose", f"{metrics['glucose_peak']:.0f} mg/dL")
            
            with col2:
                st.metric("Peak Insulin", f"{metrics['insulin_peak']:.0f} μU/mL")
                st.metric("Glucose AUC", f"{metrics['glucose_auc']:.0f}")
            
            with col3:
                st.metric("Insulinogenic Index", f"{metrics['insulinogenic_index']:.2f}")
                st.metric("Disposition Index", f"{metrics['disposition_index']:.2f}")
    
    # Test comparison table
    if len(st.session_state.test_data) > 1:
        st.markdown("### Test Comparison Summary")
        
        comparison_data = []
        for test_id, test_data in st.session_state.test_data.items():
            glucose = test_data['glucose']
            insulin = test_data['insulin']
            glucotype, _ = calculate_glucotype(glucose)
            kraft_type, _ = calculate_kraft_type(insulin, glucose)
            metrics = calculate_metrics(glucose, insulin)
            
            comparison_data.append({
                'Date': test_data['date'],
                'Glucotype': glucotype,
                'Kraft Type': kraft_type,
                'HOMA-IR': round(metrics['homa_ir'], 2),
                'Matsuda': round(metrics['matsuda'], 1),
                'Peak Glucose': round(metrics['glucose_peak'], 0),
                'Peak Insulin': round(metrics['insulin_peak'], 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Export all tests
        st.markdown("### Export All Tests")
        st.download_button(
            "Download All Tests (CSV)",
            comparison_df.to_csv(index=False),
            f"ogtt_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    st.info("Enter test data above or upload a CSV file to begin analysis")

# CSV Template Download
st.markdown("### CSV Template")
template_data = {
    'Date': ['2024-01-15', '2024-02-20'],
    'Glucose_0': [88, 92],
    'Glucose_30': [119, 135],
    'Glucose_60': [92, 110],
    'Glucose_90': [80, 85],
    'Insulin_0': [8, 12],
    'Insulin_30': [50, 65],
    'Insulin_60': [70, 85],
    'Insulin_90': [40, 45]
}
template_df = pd.DataFrame(template_data)

st.download_button(
    "Download CSV Template",
    template_df.to_csv(index=False),
    "ogtt_template.csv",
    "text/csv",
    help="Download this template to see the expected CSV format"
)

# Simple disclaimer
st.markdown("---")
st.markdown("*This tool is for educational purposes. Consult healthcare professionals for medical advice.*")