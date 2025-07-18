import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional

# Page configuration
st.set_page_config(
    page_title="Flexible OGTT Analyzer",
    page_icon="🩺",
    layout="wide"
)

# Clean, minimal styling matching original
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Avenir:wght@400;500&display=swap');

.main > div {
    padding-top: 1rem;
}

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

.test-section {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

div[data-testid="stNumberInput"] > div > div > input {
    border-radius: 6px;
    border: 1px solid #e8e8e8;
    padding: 4px;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #f0f0f0;
    border-radius: 8px;
    padding: 16px;
}

.timepoint-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0.5rem 0;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 6px;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'test_data' not in st.session_state:
    st.session_state.test_data = {}

# Flexible OGTT Analysis Functions
class FlexibleOGTTAnalyzer:
    
    @staticmethod
    def calculate_glucotype(times: List[float], glucose_vals: List[float]) -> Tuple[str, str]:
        """Calculate glucotype from flexible timepoints"""
        if len(times) < 2:
            return "Insufficient Data", "concern"
        
        # Sort by time
        sorted_pairs = sorted(zip(times, glucose_vals))
        sorted_times, sorted_glucose = zip(*sorted_pairs)
        
        baseline = sorted_glucose[0]
        peak_glucose = max(sorted_glucose)
        peak_time = sorted_times[sorted_glucose.index(peak_glucose)]
        final_glucose = sorted_glucose[-1]
        
        # Classification logic
        if peak_glucose < baseline + 20:
            return "Flat Responder", "concern"
        elif peak_time <= 30:
            if final_glucose <= baseline + 15:
                return "Normal Monophasic", "normal"
            else:
                return "Sustained Elevation", "warning"
        elif peak_time > 60:
            return "Delayed Peak", "warning"
        else:
            return "Standard Response", "normal"
    
    @staticmethod
    def calculate_kraft_type(times: List[float], insulin_vals: List[float]) -> Tuple[str, str]:
        """Calculate Kraft pattern from flexible timepoints"""
        if len(times) < 2:
            return "Insufficient Data", "concern"
        
        # Sort by time
        sorted_pairs = sorted(zip(times, insulin_vals))
        sorted_times, sorted_insulin = zip(*sorted_pairs)
        
        baseline_insulin = sorted_insulin[0]
        peak_insulin = max(sorted_insulin)
        peak_time = sorted_times[sorted_insulin.index(peak_insulin)]
        final_insulin = sorted_insulin[-1] if len(sorted_insulin) > 2 else peak_insulin
        
        # Classification
        if baseline_insulin >= 25:
            return "Type IV - High Fasting", "concern"
        elif peak_insulin < 30:
            return "Type V - Low Response", "concern"
        elif peak_time <= 30 and final_insulin < baseline_insulin + 15:
            return "Type I - Normal", "normal"
        elif peak_time > 60:
            return "Type II - Delayed", "warning"
        elif final_insulin >= 60:
            return "Type III - Sustained", "warning"
        else:
            return "Type I/II - Borderline", "warning"
    
    @staticmethod
    def calculate_metrics(times: List[float], glucose_vals: List[float], insulin_vals: List[float]) -> Dict:
        """Calculate metabolic metrics from flexible timepoints"""
        if len(times) < 2:
            return {}
        
        # Sort data
        glucose_pairs = sorted(zip(times, glucose_vals))
        insulin_pairs = sorted(zip(times, insulin_vals))
        
        g_times, g_values = zip(*glucose_pairs)
        i_times, i_values = zip(*insulin_pairs)
        
        g0, i0 = g_values[0], i_values[0]
        
        metrics = {
            'glucose_peak': max(g_values),
            'insulin_peak': max(i_values),
            'timepoint_count': len(times),
            'time_range': f"{min(times):.0f}-{max(times):.0f} min"
        }
        
        # HOMA-IR (if baseline available)
        if g0 > 0 and i0 > 0:
            metrics['homa_ir'] = (g0 * i0) / 405
        
        # Matsuda Index (if multiple points)
        if len(g_values) >= 2:
            mean_glucose = np.mean(g_values)
            mean_insulin = np.mean(i_values)
            if all(v > 0 for v in [g0, i0, mean_glucose, mean_insulin]):
                metrics['matsuda'] = 10000 / math.sqrt(g0 * i0 * mean_glucose * mean_insulin)
        
        # AUC using trapezoidal rule
        if len(g_values) > 1:
            metrics['glucose_auc'] = np.trapz(g_values, g_times)
            metrics['insulin_auc'] = np.trapz(i_values, i_times)
        
        # Insulinogenic index (if 30-min point exists or can be interpolated)
        if 30 in times:
            idx_30 = list(times).index(30)
            g30, i30 = glucose_vals[idx_30], insulin_vals[idx_30]
            if g30 > g0:
                metrics['insulinogenic_index'] = (i30 - i0) / (g30 - g0)
        elif len(times) > 2 and max(times) >= 30:
            g30 = np.interp(30, g_times, g_values)
            i30 = np.interp(30, i_times, i_values)
            if g30 > g0:
                metrics['insulinogenic_index'] = (i30 - i0) / (g30 - g0)
        
        return metrics

# Enhanced plotting function with all visual controls and shading
def create_ogtt_plot(title: str, y_label: str, is_glucose: bool = True) -> go.Figure:
    """Create OGTT plot with original styling and full visual controls"""
    fig = go.Figure()
    
    # Original color scheme
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A2B8', '#FD7E14', '#6C757D', '#DC3545', '#20C997']
    
    # Get settings from session state
    line_width = st.session_state.get('line_width', 4)
    marker_size = st.session_state.get('marker_size', 10)
    show_reference = st.session_state.get('show_reference', True)
    show_shading = st.session_state.get('show_shading', False)
    show_confidence = st.session_state.get('show_confidence', False)
    show_legend = st.session_state.get('show_legend', True)
    reference_color = st.session_state.get('reference_color', '#1cb164')
    reference_opacity = st.session_state.get('reference_opacity', 0.5)
    reference_width = st.session_state.get('reference_width', 3)
    title_font_size = st.session_state.get('title_font_size', 32)
    axis_title_size = st.session_state.get('axis_title_size', 20)
    axis_tick_size = st.session_state.get('axis_tick_size', 16)
    test_visibility = st.session_state.get('test_visibility', {})
    
    # Reference data
    if is_glucose:
        ref_times = [0, 30, 60, 90]
        ref_values = [90, 140, 120, 95]
    else:
        ref_times = [0, 30, 60, 90]
        ref_values = [6, 40, 30, 20]
    
    # Add confidence intervals if enabled
    if show_confidence and show_reference:
        # Simple confidence interval calculation
        if is_glucose:
            relative_se = 0.12  # 12% standard error for glucose
        else:
            relative_se = 0.20  # 20% standard error for insulin
        
        upper_bound = [v * (1 + 1.96 * relative_se) for v in ref_values]
        lower_bound = [max(0, v * (1 - 1.96 * relative_se)) for v in ref_values]
        
        fig.add_trace(go.Scatter(
            x=ref_times + ref_times[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(180, 180, 180, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Add reference line
    if show_reference:
        fig.add_trace(go.Scatter(
            x=ref_times,
            y=ref_values,
            mode='lines',
            name='Normal Pattern',
            line=dict(color=reference_color, width=reference_width, dash='dot'),
            opacity=reference_opacity
        ))
    
    # Add shading for all tests if enabled
    if show_shading and show_reference:
        for i, (test_id, test_data) in enumerate(st.session_state.test_data.items()):
            if test_id not in test_visibility or test_visibility[test_id]:
                test_times = test_data.get('times', [])
                test_glucose = test_data.get('glucose', [])
                test_insulin = test_data.get('insulin', [])
                test_color = colors[i % len(colors)]
                
                test_values = test_glucose if is_glucose else test_insulin
                
                # Create valid pairs
                valid_pairs = []
                for j in range(min(len(test_times), len(test_values))):
                    t, v = test_times[j], test_values[j]
                    if t is not None and v is not None:
                        valid_pairs.append((float(t), float(v)))
                
                if len(valid_pairs) >= 2:
                    valid_pairs.sort()
                    valid_times, valid_values = zip(*valid_pairs)
                    
                    # Create shading between user data and reference
                    for j in range(len(valid_times) - 1):
                        x1, x2 = valid_times[j], valid_times[j+1]
                        y1_user, y2_user = valid_values[j], valid_values[j+1]
                        
                        # Interpolate reference values
                        if x1 in ref_times and x2 in ref_times:
                            y1_ref = ref_values[ref_times.index(x1)]
                            y2_ref = ref_values[ref_times.index(x2)]
                        else:
                            import numpy as np
                            y1_ref = np.interp(x1, ref_times, ref_values)
                            y2_ref = np.interp(x2, ref_times, ref_values)
                        
                        # Determine if user values are above or below reference
                        user_above = (y1_user + y2_user) / 2 > (y1_ref + y2_ref) / 2
                        
                        # Choose colors
                        if user_above:
                            fill_color = f'rgba(231, 76, 60, 0.08)'  # Light red for above
                        else:
                            fill_color = f'rgba(46, 204, 113, 0.08)'  # Light green for below
                        
                        # Create the shaded area
                        x_fill = [x1, x2, x2, x1]
                        if user_above:
                            y_fill = [y1_ref, y2_ref, y2_user, y1_user]
                        else:
                            y_fill = [y1_user, y2_user, y2_ref, y1_ref]
                        
                        fig.add_trace(go.Scatter(
                            x=x_fill, y=y_fill, 
                            fill='toself', 
                            fillcolor=fill_color,
                            line=dict(color='rgba(0,0,0,0)'), 
                            showlegend=False, 
                            hoverinfo='skip'
                        ))
    
    # Add test data
    for i, (test_id, test_data) in enumerate(st.session_state.test_data.items()):
        if test_id not in test_visibility or test_visibility[test_id]:
            test_times = test_data.get('times', [])
            test_glucose = test_data.get('glucose', [])
            test_insulin = test_data.get('insulin', [])
            test_color = colors[i % len(colors)]
            test_date = test_data.get('date', f'Test {test_id}')
            
            # Choose the appropriate values
            test_values = test_glucose if is_glucose else test_insulin
            
            # Create valid pairs
            valid_pairs = []
            for j in range(min(len(test_times), len(test_values))):
                t, v = test_times[j], test_values[j]
                if t is not None and v is not None:
                    try:
                        valid_pairs.append((float(t), float(v)))
                    except (ValueError, TypeError):
                        continue
            
            # Skip if insufficient data
            if len(valid_pairs) < 2:
                continue
            
            # Sort by time and extract values
            valid_pairs.sort(key=lambda x: x[0])
            valid_times, valid_values = zip(*valid_pairs)
            
            fig.add_trace(go.Scatter(
                x=valid_times,
                y=valid_values,
                mode='lines+markers',
                name=test_date,
                line=dict(color=test_color, width=line_width),
                marker=dict(size=marker_size, color=test_color, line=dict(color='white', width=2))
            ))
    
    # Original layout styling with typography controls
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
            showgrid=True, gridcolor='#ecf0f1', gridwidth=1
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(family="Cormorant Garamond, serif", size=axis_title_size, color='#34495e')),
            tickfont=dict(family="Avenir, sans-serif", size=axis_tick_size, color='#7f8c8d'),
            showgrid=True, gridcolor='#ecf0f1', gridwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=40, l=60, r=40),
        font=dict(family="Avenir, sans-serif")
    )
    
    return fig

# Header
st.markdown("""
<div class="simple-header">
    <h1>Flexible OGTT Analyzer</h1>
    <p>Multi-Test OGTT Analysis Supporting Any Timepoint Configuration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
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
                colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD']
                
                for i, row in df.iterrows():
                    test_id = i + 1
                    color = colors[i % len(colors)]
                    
                    # Flexible column detection
                    times, glucose, insulin = [], [], []
                    
                    # Method 1: Look for numbered columns (glucose_0, glucose_30, etc.)
                    timepoints = set()
                    for col in df.columns:
                        if any(pattern in col.lower() for pattern in ['_0', '_15', '_30', '_45', '_60', '_90', '_120']):
                            try:
                                timepoint = int(col.split('_')[-1])
                                timepoints.add(timepoint)
                            except:
                                pass
                    
                    timepoints = sorted(timepoints)
                    for tp in timepoints:
                        g_col = next((col for col in df.columns if f'glucose_{tp}' in col.lower()), None)
                        i_col = next((col for col in df.columns if f'insulin_{tp}' in col.lower()), None)
                        
                        if g_col and i_col and pd.notna(row[g_col]) and pd.notna(row[i_col]):
                            times.append(float(tp))
                            glucose.append(float(row[g_col]))
                            insulin.append(float(row[i_col]))
                    
                    # Fallback to default if no data found
                    if not times:
                        times = [0, 30, 60, 90]
                        glucose = [88, 119, 92, 80]
                        insulin = [8, 50, 70, 40]
                    
                    # Date handling
                    date_str = row.get('Date', datetime.now().strftime('%Y-%m-%d'))
                    try:
                        parsed_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
                    except:
                        parsed_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                    
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
    
    # Manual test input
    num_tests = st.number_input("Number of tests:", min_value=1, max_value=10, value=1, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Tests"):
            st.session_state.test_data = {}
            st.rerun()
    with col2:
        if st.button("Add Sample Data"):
            # Add sample data with different timepoint configurations
            sample_data = {
                1: {'date': '2024-01-15', 'times': [0, 30, 60, 90], 
                    'glucose': [88, 119, 92, 80], 'insulin': [8, 50, 70, 40], 'color': '#2E86C1'},
                2: {'date': '2024-02-20', 'times': [0, 30, 120], 
                    'glucose': [92, 135, 85], 'insulin': [12, 65, 45], 'color': '#E74C3C'}
            }
            # Only add sample data that fits within current num_tests
            for test_id in range(1, min(3, num_tests + 1)):
                if test_id in sample_data:
                    st.session_state.test_data[test_id] = sample_data[test_id]
            st.rerun()
    
    # Chart appearance with all original controls
    with st.expander("Chart Appearance", expanded=True):
        line_width = st.slider("Line width", 1, 10, 4)
        marker_size = st.slider("Marker size", 4, 20, 10)
        show_shading = st.checkbox("Show comparative shading", value=False)
        show_reference = st.checkbox("Show reference line", value=True)
        show_confidence = st.checkbox("Show confidence intervals", value=False)
        show_legend = st.checkbox("Show legend", value=True)
        
        # Store in session state
        st.session_state.line_width = line_width
        st.session_state.marker_size = marker_size
        st.session_state.show_reference = show_reference
        st.session_state.show_legend = show_legend
        st.session_state.show_shading = show_shading
        st.session_state.show_confidence = show_confidence
    
    # Test visibility toggles
    if st.session_state.test_data:
        st.subheader("Test Visibility")
        test_visibility = {}
        for test_id in st.session_state.test_data.keys():
            test_date = st.session_state.test_data[test_id].get('date', f'Test {test_id}')
            test_visibility[test_id] = st.checkbox(f"Show {test_date}", value=True, key=f"vis_{test_id}")
        st.session_state.test_visibility = test_visibility
    else:
        test_visibility = {}
        st.session_state.test_visibility = {}
    
    # Reference line details
    with st.expander("Reference Line Details"):
        reference_color = st.color_picker("Reference line color", "#1cb164")
        reference_opacity = st.slider("Reference line opacity", 0.1, 1.0, 0.5, 0.1)
        reference_width = st.slider("Reference line width", 1, 10, 3)
        
        # Store in session state
        st.session_state.reference_color = reference_color
        st.session_state.reference_opacity = reference_opacity
        st.session_state.reference_width = reference_width
    
    # Annotations
    with st.expander("Annotations"):
        show_annotations = st.checkbox("Show value annotations", value=True)
        annotation_size = st.slider("Text size", 8, 48, 16)
        annotation_bold = st.checkbox("Bold text", value=True)
        annotation_color = st.color_picker("Text color", "#2c3e50")
        show_annotation_bg = st.checkbox("Show annotation background", value=False)
        if show_annotation_bg:
            annotation_bg_color = st.color_picker("Background color", "#ffffff")
            annotation_border_color = st.color_picker("Border color", "#cccccc")
        else:
            annotation_bg_color = "rgba(255, 255, 255, 0)"
            annotation_border_color = "rgba(0, 0, 0, 0)"
        
        # Store in session state
        st.session_state.show_annotations = show_annotations
        st.session_state.annotation_size = annotation_size
        st.session_state.annotation_bold = annotation_bold
        st.session_state.annotation_color = annotation_color
        st.session_state.annotation_bg_color = annotation_bg_color
        st.session_state.annotation_border_color = annotation_border_color
    
    # Typography
    with st.expander("Typography"):
        title_font_size = st.slider("Chart title size", 12, 72, 32)
        axis_title_size = st.slider("Axis title size", 8, 48, 20)
        axis_tick_size = st.slider("Axis tick size", 6, 36, 16)
        
        # Store in session state
        st.session_state.title_font_size = title_font_size
        st.session_state.axis_title_size = axis_title_size
        st.session_state.axis_tick_size = axis_tick_size
    
    # Export settings
    with st.expander("Export Settings"):
        download_width = st.slider("Width (px)", 800, 4000, 1200, step=200)
        download_height = st.slider("Height (px)", 400, 3000, 550, step=50)
        download_scale = st.slider("Scale multiplier", 1, 10, 1)

# Main content
st.markdown("### Test Data Entry")

# Color palette for tests
colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A2B8', '#FD7E14', '#6C757D', '#DC3545', '#20C997']

# Create input sections for each test
for i in range(num_tests):
    test_id = i + 1
    color = colors[i % len(colors)]
    
    with st.expander(f"Test {test_id}", expanded=True):
        
        # Simple date text input
        default_date = st.session_state.test_data.get(test_id, {}).get('date', datetime.now().strftime('%Y-%m-%d'))
        test_date = st.text_input(
            "Date:",
            value=default_date,
            key=f"date_{test_id}",
            placeholder="YYYY-MM-DD"
        )
        
        # Simple comma-separated input (clean, no presets)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Time Points (min)**")
            time_input = st.text_input(
                "", 
                key=f"times_{test_id}",
                placeholder="0, 30, 60, 90"
            )
            
        with col2:
            st.markdown("**Glucose (mg/dL)**")
            glucose_input = st.text_input(
                "", 
                key=f"glucose_{test_id}",
                placeholder="88, 119, 92, 80"
            )
            
        with col3:
            st.markdown("**Insulin (μU/mL)**")
            insulin_input = st.text_input(
                "", 
                key=f"insulin_{test_id}",
                placeholder="8, 50, 70, 40"
            )
        
        # Parse and auto-save data
        try:
            # Only process if user has entered data
            if time_input or glucose_input or insulin_input:
                times = [float(x.strip()) for x in time_input.split(',') if x.strip()] if time_input else []
                glucose_vals = [float(x.strip()) for x in glucose_input.split(',') if x.strip()] if glucose_input else []
                insulin_vals = [float(x.strip()) for x in insulin_input.split(',') if x.strip()] if insulin_input else []
                
                # Save the data as entered
                st.session_state.test_data[test_id] = {
                    'date': test_date,
                    'times': times,
                    'glucose': glucose_vals,
                    'insulin': insulin_vals,
                    'color': color
                }
                
                # Show preview if we have valid data
                if times and glucose_vals and insulin_vals:
                    # Find the minimum length for preview
                    min_len = min(len(times), len(glucose_vals), len(insulin_vals))
                    if min_len >= 2:
                        st.markdown("**Preview:**")
                        preview_df = pd.DataFrame({
                            'Time (min)': times[:min_len],
                            'Glucose (mg/dL)': glucose_vals[:min_len],
                            'Insulin (μU/mL)': insulin_vals[:min_len]
                        })
                        st.dataframe(preview_df, use_container_width=True)
                        
                        if min_len < max(len(times), len(glucose_vals), len(insulin_vals)):
                            st.info(f"ℹ️ Using first {min_len} values (shortest list). Make sure all lists have the same length for complete data.")
                    elif min_len > 0:
                        st.warning(f"⚠️ Need at least 2 matching timepoints for analysis.")
            else:
                # Initialize with defaults if no input provided and not already set
                if test_id not in st.session_state.test_data:
                    st.session_state.test_data[test_id] = {
                        'date': test_date,
                        'times': [0, 30, 60, 90],
                        'glucose': [88, 119, 92, 80],
                        'insulin': [8, 50, 70, 40],
                        'color': color
                    }
            
        except ValueError as e:
            st.error(f"❌ Error parsing values. Please use comma-separated numbers only.")
            # Keep defaults if parsing fails
            if test_id not in st.session_state.test_data:
                st.session_state.test_data[test_id] = {
                    'date': test_date,
                    'times': [0, 30, 60, 90],
                    'glucose': [88, 119, 92, 80],
                    'insulin': [8, 50, 70, 40],
                    'color': color
                }

# Clean up test_data when reducing number of tests
existing_test_ids = list(st.session_state.test_data.keys())
for test_id in existing_test_ids:
    if test_id > num_tests:
        del st.session_state.test_data[test_id]

# Analysis and visualization
if st.session_state.test_data:
    
    # Create plots with annotations
    st.markdown("### Glucose Response - All Tests")
    fig1 = create_ogtt_plot('Glucose Response', 'Glucose (mg/dL)', is_glucose=True)
    
    # Add annotations if enabled
    if st.session_state.get('show_annotations', True):
        show_annotations = st.session_state.get('show_annotations', True)
        annotation_size = st.session_state.get('annotation_size', 16)
        annotation_bold = st.session_state.get('annotation_bold', True)
        annotation_color = st.session_state.get('annotation_color', '#2c3e50')
        annotation_bg_color = st.session_state.get('annotation_bg_color', 'rgba(255, 255, 255, 0)')
        annotation_border_color = st.session_state.get('annotation_border_color', 'rgba(0, 0, 0, 0)')
        test_visibility = st.session_state.get('test_visibility', {})
        
        for test_id, test_data in st.session_state.test_data.items():
            if test_id not in test_visibility or test_visibility[test_id]:
                test_times = test_data.get('times', [])
                test_glucose = test_data.get('glucose', [])
                
                valid_pairs = [(t, g) for t, g in zip(test_times, test_glucose) 
                             if t is not None and g is not None]
                for t, g in valid_pairs:
                    try:
                        text_content = f"<b>{g:.0f}</b>" if annotation_bold else f"{g:.0f}"
                        fig1.add_annotation(
                            x=float(t), y=float(g), text=text_content, 
                            showarrow=False, 
                            yshift=25,
                            font=dict(family="Avenir, sans-serif", size=annotation_size, color=annotation_color),
                            bgcolor=annotation_bg_color,
                            bordercolor=annotation_border_color,
                            borderwidth=1 if 'rgba(255, 255, 255, 0)' not in annotation_bg_color else 0,
                            borderpad=4 if 'rgba(255, 255, 255, 0)' not in annotation_bg_color else 0
                        )
                    except (ValueError, TypeError):
                        continue
    
    st.plotly_chart(fig1, use_container_width=True, config={
        'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage'],
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'glucose_comparison',
            'height': download_height, 'width': download_width, 'scale': download_scale
        }
    })
    
    st.markdown("### Insulin Response - All Tests")
    fig2 = create_ogtt_plot('Insulin Response', 'Insulin (μU/mL)', is_glucose=False)
    
    # Add annotations for insulin if enabled
    if st.session_state.get('show_annotations', True):
        for test_id, test_data in st.session_state.test_data.items():
            if test_id not in test_visibility or test_visibility[test_id]:
                test_times = test_data.get('times', [])
                test_insulin = test_data.get('insulin', [])
                
                valid_pairs = [(t, i) for t, i in zip(test_times, test_insulin) 
                             if t is not None and i is not None]
                for t, i in valid_pairs:
                    try:
                        text_content = f"<b>{i:.0f}</b>" if annotation_bold else f"{i:.0f}"
                        fig2.add_annotation(
                            x=float(t), y=float(i), text=text_content, 
                            showarrow=False, 
                            yshift=25,
                            font=dict(family="Avenir, sans-serif", size=annotation_size, color=annotation_color),
                            bgcolor=annotation_bg_color,
                            bordercolor=annotation_border_color,
                            borderwidth=1 if 'rgba(255, 255, 255, 0)' not in annotation_bg_color else 0,
                            borderpad=4 if 'rgba(255, 255, 255, 0)' not in annotation_bg_color else 0
                        )
                    except (ValueError, TypeError):
                        continue
    
    st.plotly_chart(fig2, use_container_width=True, config={
        'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage'],
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'insulin_comparison',
            'height': download_height, 'width': download_width, 'scale': download_scale
        }
    })
    
    # Analysis for each test
    st.markdown("### Individual Test Analysis")
    
    analyzer = FlexibleOGTTAnalyzer()
    
    for test_id, test_data in st.session_state.test_data.items():
        times = test_data.get('times', [])
        glucose = test_data.get('glucose', [])
        insulin = test_data.get('insulin', [])
        test_date = test_data.get('date', f'Test {test_id}')
        
        # Validate data - handle flexible lengths gracefully
        valid_glucose_pairs = [(t, g) for t, g in zip(times, glucose) 
                              if t is not None and g is not None]
        valid_insulin_pairs = [(t, i) for t, i in zip(times, insulin) 
                              if t is not None and i is not None]
        
        if len(valid_glucose_pairs) < 2 or len(valid_insulin_pairs) < 2:
            st.warning(f"⚠️ {test_date}: Need at least 2 valid timepoints for analysis")
            continue
        
        # Extract times and values for analysis
        glucose_times, glucose_values = zip(*valid_glucose_pairs)
        insulin_times, insulin_values = zip(*valid_insulin_pairs)
        
        # Use the longer dataset for analysis, but warn if they don't match
        if len(glucose_times) != len(insulin_times):
            st.info(f"ℹ️ {test_date}: Glucose has {len(glucose_times)} points, Insulin has {len(insulin_times)} points")
        
        # Calculate results using the available data
        glucotype, gluco_status = analyzer.calculate_glucotype(glucose_times, glucose_values)
        kraft_type, kraft_status = analyzer.calculate_kraft_type(insulin_times, insulin_values)
        
        # For metrics, use matching timepoints only
        matching_pairs = []
        for t in set(glucose_times) & set(insulin_times):
            g_idx = glucose_times.index(t)
            i_idx = insulin_times.index(t)
            matching_pairs.append((t, glucose_values[g_idx], insulin_values[i_idx]))
        
        if matching_pairs:
            matching_pairs.sort()
            match_times, match_glucose, match_insulin = zip(*matching_pairs)
            metrics = analyzer.calculate_metrics(match_times, match_glucose, match_insulin)
        else:
            metrics = {}
        
        # Display results
        st.markdown(f"#### {test_date} ({len(valid_glucose_pairs)} glucose, {len(valid_insulin_pairs)} insulin points)")
        
        if len(valid_glucose_pairs) < 4 and len(valid_insulin_pairs) < 4:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Limited Data:</strong> This test has fewer than 4 timepoints. 
                Analysis may be less comprehensive than standard OGTT protocols.
            </div>
            """, unsafe_allow_html=True)
        
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
            if metrics.get('homa_ir'):
                homa_status = "normal" if metrics['homa_ir'] < 1.0 else "warning" if metrics['homa_ir'] < 2.5 else "concern"
                st.markdown(f"""
                <div class="result-box {homa_status}">
                    <strong>HOMA-IR</strong><br>
                    {metrics['homa_ir']:.2f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box concern">
                    <strong>HOMA-IR</strong><br>
                    N/A (need baseline)
                </div>
                """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if metrics.get('matsuda'):
                st.metric("Matsuda Index", f"{metrics['matsuda']:.1f}")
            st.metric("Peak Glucose", f"{metrics['glucose_peak']:.0f} mg/dL")
        
        with col2:
            st.metric("Peak Insulin", f"{metrics['insulin_peak']:.0f} μU/mL")
            if metrics.get('glucose_auc'):
                st.metric("Glucose AUC", f"{metrics['glucose_auc']:.0f}")
        
        with col3:
            if metrics.get('insulinogenic_index'):
                st.metric("Insulinogenic Index", f"{metrics['insulinogenic_index']:.2f}")
            st.metric("Timepoints", metrics['timepoint_count'])
    
    # Test comparison table
    if len(st.session_state.test_data) > 1:
        st.markdown("### Test Comparison Summary")
        
        comparison_data = []
        for test_id, test_data in st.session_state.test_data.items():
            times = test_data.get('times', [])
            glucose = test_data.get('glucose', [])
            insulin = test_data.get('insulin', [])
            
            # Validate data
            valid_glucose_pairs = [(t, g) for t, g in zip(times, glucose) 
                                  if t is not None and g is not None]
            valid_insulin_pairs = [(t, i) for t, i in zip(times, insulin) 
                                  if t is not None and i is not None]
            
            if len(valid_glucose_pairs) < 2 or len(valid_insulin_pairs) < 2:
                continue
            
            glucose_times, glucose_values = zip(*valid_glucose_pairs)
            insulin_times, insulin_values = zip(*valid_insulin_pairs)
            
            glucotype, _ = analyzer.calculate_glucotype(glucose_times, glucose_values)
            kraft_type, _ = analyzer.calculate_kraft_type(insulin_times, insulin_values)
            
            # For metrics, use matching timepoints
            matching_pairs = []
            for t in set(glucose_times) & set(insulin_times):
                g_idx = glucose_times.index(t)
                i_idx = insulin_times.index(t)
                matching_pairs.append((t, glucose_values[g_idx], insulin_values[i_idx]))
            
            if matching_pairs:
                matching_pairs.sort()
                match_times, match_glucose, match_insulin = zip(*matching_pairs)
                metrics = analyzer.calculate_metrics(match_times, match_glucose, match_insulin)
            else:
                metrics = {'glucose_peak': max(glucose_values), 'insulin_peak': max(insulin_values)}
            
            comparison_data.append({
                'Date': test_data.get('date', f'Test {test_id}'),
                'Glucose Points': len(valid_glucose_pairs),
                'Insulin Points': len(valid_insulin_pairs),
                'Time Range': f"{min(glucose_times):.0f}-{max(glucose_times):.0f} min" if glucose_times else 'N/A',
                'Glucotype': glucotype,
                'Kraft Type': kraft_type,
                'HOMA-IR': round(metrics.get('homa_ir', 0), 2) if metrics.get('homa_ir') else 'N/A',
                'Matsuda': round(metrics.get('matsuda', 0), 1) if metrics.get('matsuda') else 'N/A',
                'Peak Glucose': round(metrics.get('glucose_peak', 0), 0),
                'Peak Insulin': round(metrics.get('insulin_peak', 0), 0)
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Export comparison
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                "Download Comparison (CSV)",
                csv_data,
                f"ogtt_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

else:
    st.info("Enter test data above or upload a CSV file to begin analysis")

# CSV Template Download
st.markdown("### CSV Template Examples")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Standard Format")
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
    st.dataframe(template_df, use_container_width=True)
    
    st.download_button(
        "Download Standard Template",
        template_df.to_csv(index=False),
        "ogtt_standard_template.csv",
        "text/csv"
    )

with col2:
    st.markdown("#### Extended Format")
    extended_template = {
        'Date': ['2024-01-15', '2024-02-20'],
        'Glucose_0': [88, 92],
        'Glucose_15': [105, 120],
        'Glucose_30': [119, 135],
        'Glucose_60': [92, 110],
        'Glucose_90': [80, 85],
        'Glucose_120': [75, 80],
        'Insulin_0': [8, 12],
        'Insulin_15': [35, 45],
        'Insulin_30': [50, 65],
        'Insulin_60': [70, 85],
        'Insulin_90': [40, 45],
        'Insulin_120': [25, 30]
    }
    extended_df = pd.DataFrame(extended_template)
    st.dataframe(extended_df, use_container_width=True)
    
    st.download_button(
        "Download Extended Template",
        extended_df.to_csv(index=False),
        "ogtt_extended_template.csv",
        "text/csv"
    )

# Help section
with st.expander("Help & Information", expanded=False):
    st.markdown("""
    ## Flexible OGTT Analyzer Help
    
    ### Supported Timepoint Formats
    
    **Standard Protocols:**
    - **Clinical**: 0, 60, 120 minutes (minimal)
    - **Standard**: 0, 30, 60, 90 minutes (recommended)
    - **Extended**: 0, 30, 60, 90, 120 minutes (comprehensive)
    - **Research**: 0, 15, 30, 60, 90, 120 minutes (detailed)
    
    **Custom Protocols:**
    - Any combination of timepoints from 0-480 minutes
    - Minimum 2 timepoints required for analysis
    - Baseline (t=0) strongly recommended for complete metrics
    
    ### Analysis Features
    
    **Pattern Recognition:**
    - **Glucotype**: Glucose response pattern classification
    - **Kraft Pattern**: Insulin response classification (Types I-V)
    
    **Metabolic Metrics:**
    - **HOMA-IR**: Insulin resistance (requires baseline values)
    - **Matsuda Index**: Insulin sensitivity (requires multiple points)
    - **AUC**: Area under curve using trapezoidal integration
    - **Peak Values**: Maximum glucose and insulin responses
    - **Insulinogenic Index**: Early insulin response (when 30-min available)
    
    ### CSV Upload Formats
    
    **Method 1 - Numbered Columns:**
    Use column names like `Glucose_0`, `Glucose_30`, `Insulin_0`, `Insulin_30`, etc.
    The number after the underscore represents the time in minutes.
    
    **Method 2 - Any Timepoints:**
    You can use any timepoint numbers: `Glucose_15`, `Glucose_45`, `Insulin_75`, etc.
    
    ### Normal Ranges
    
    - **Fasting Glucose**: 70-99 mg/dL
    - **2-hour Glucose**: <140 mg/dL
    - **HOMA-IR**: <1.0 (excellent), 1.0-2.5 (good), >2.5 (resistance)
    - **Matsuda Index**: >4.0 (good sensitivity), 2.5-4.0 (moderate), <2.5 (low)
    
    ### Data Quality
    
    - **Minimum**: 2 timepoints (basic analysis only)
    - **Recommended**: 4+ timepoints (comprehensive analysis)
    - **Optimal**: Include baseline (t=0) for complete metrics
    - **Range Check**: Glucose 40-400 mg/dL, Insulin 1-300 μU/mL
    
    ### Important Notes
    
    - This tool is for educational and research purposes only
    - Always consult healthcare professionals for medical interpretation
    - Missing timepoints may affect analysis accuracy
    - Consider individual patient factors and clinical context
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem 0;">
    💡 <strong>Flexible OGTT Analyzer</strong> | Supporting Any Timepoint Configuration<br>
    🏥 For educational and research purposes only | Always consult healthcare professionals
</div>
""", unsafe_allow_html=True)