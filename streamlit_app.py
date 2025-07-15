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

/* Compact columns */
.compact-input {
    padding: 0.25rem;
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
</style>
""", unsafe_allow_html=True)

# Compact header
st.markdown("""
<div class="simple-header">
    <h1>OGTT Analyzer</h1>
    <p>Oral Glucose Tolerance Test Analysis</p>
</div>
""", unsafe_allow_html=True)

# Reorganized sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Data input first (most important)
    st.subheader("📊 Data Input")
    input_method = st.radio("Choose input method:", ["Manual Entry", "CSV Upload", "Paste Data"])
    
    # Chart appearance (commonly used)
    with st.expander("🎨 Chart Appearance", expanded=True):
        glucose_color = st.color_picker("Glucose line color", "#2E86C1")
        insulin_color = st.color_picker("Insulin line color", "#3498DB")
        line_width = st.slider("Line width", 1, 10, 5)
        marker_size = st.slider("Marker size", 4, 20, 13)
        
        show_shading = st.checkbox("Show comparative shading", value=True)
        show_reference = st.checkbox("Show reference line", value=True)
    
    # Reference line details (collapsed by default)
    with st.expander("📏 Reference Line Details"):
        reference_color = st.color_picker("Reference line color", "#1cb164")
        reference_opacity = st.slider("Reference line opacity", 0.1, 1.0, 0.5, 0.1)
        reference_width = st.slider("Reference line width", 1, 10, 3)
    
    # Annotations (commonly used)
    with st.expander("📝 Annotations", expanded=True):
        show_annotations = st.checkbox("Show value annotations", value=True)
        show_reference_annotations = st.checkbox("Show reference annotations", value=False)
        annotation_size = st.slider("Text size", 8, 48, 24)
        annotation_bold = st.checkbox("Bold text", value=True)
        annotation_color = st.color_picker("Text color", "#2c3e50")
    
    # Typography (for power users)
    with st.expander("🔤 Typography"):
        title_font_size = st.slider("Chart title size", 12, 72, 38)
        axis_title_size = st.slider("Axis title size", 8, 48, 24)
        axis_tick_size = st.slider("Axis tick size", 6, 36, 18)
    
    # Export settings (less frequently used)
    with st.expander("💾 Export Settings"):
        download_width = st.slider("Width (px)", 800, 4000, 1200, step=200)
        download_height = st.slider("Height (px)", 400, 3000, 550, step=50)
        download_scale = st.slider("Scale multiplier", 1, 10, 1)
        st.caption(f"Final: {download_width * download_scale} × {download_height * download_scale} px")
        st.caption("💡 Optimized for presentations")

# Compact data input based on method
if input_method == "Manual Entry":
    st.markdown("### Quick Entry")
    # Three text input boxes for comma-separated values
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Time Points (min)**")
        time_input = st.text_input("", value="0, 30, 60, 90", key="times")
        
    with col2:
        st.markdown("**Glucose (mg/dL)**")
        glucose_input = st.text_input("", value="88, 119, 92, 80", key="glucose")
        
    with col3:
        st.markdown("**Insulin (μU/mL)**")
        insulin_input = st.text_input("", value="8, 50, 70, 40", key="insulin")
    
    # Parse the comma-separated values
    try:
        times = [float(x.strip()) for x in time_input.split(',')]
        glucose_vals = [float(x.strip()) for x in glucose_input.split(',')]
        insulin_vals = [float(x.strip()) for x in insulin_input.split(',')]
        
        if len(times) >= 4 and len(glucose_vals) >= 4 and len(insulin_vals) >= 4:
            g0, g30, g60, g90 = glucose_vals[:4]
            i0, i30, i60, i90 = insulin_vals[:4]
            times = times[:4]  # Use first 4 time points
        else:
            st.error("Please provide at least 4 values for each measurement")
            g0, g30, g60, g90 = 88, 119, 92, 80
            i0, i30, i60, i90 = 8, 50, 70, 40
            times = [0, 30, 60, 90]
    except:
        st.error("Please enter valid numbers separated by commas")
        g0, g30, g60, g90 = 88, 119, 92, 80
        i0, i30, i60, i90 = 8, 50, 70, 40
        times = [0, 30, 60, 90]

elif input_method == "CSV Upload":
    st.subheader("Upload CSV File")
    st.write("Expected format: columns for Time, Glucose, Insulin")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(df)
        
        # Try to extract values (assuming standard format)
        if len(df) >= 4:
            glucose_vals = df['Glucose'].tolist()[:4] if 'Glucose' in df.columns else [88, 119, 92, 80]
            insulin_vals = df['Insulin'].tolist()[:4] if 'Insulin' in df.columns else [8, 50, 70, 40]
            times = df['Time'].tolist()[:4] if 'Time' in df.columns else [0, 30, 60, 90]
            g0, g30, g60, g90 = glucose_vals
            i0, i30, i60, i90 = insulin_vals
        else:
            st.error("CSV must contain at least 4 time points")
            g0, g30, g60, g90 = 88, 119, 92, 80
            i0, i30, i60, i90 = 8, 50, 70, 40
            times = [0, 30, 60, 90]
    else:
        # Default values
        g0, g30, g60, g90 = 88, 119, 92, 80
        i0, i30, i60, i90 = 8, 50, 70, 40
        times = [0, 30, 60, 90]

else:  # Paste Data
    st.subheader("Paste Tab-Separated Data")
    st.write("Format: Time | Glucose | Insulin (one row per time point)")
    
    pasted_data = st.text_area("Paste your data here:", height=100)
    
    if pasted_data:
        try:
            lines = pasted_data.strip().split('\n')
            times = []
            glucose_vals = []
            insulin_vals = []
            
            for line in lines:
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) >= 3:
                    times.append(float(parts[0]))
                    glucose_vals.append(float(parts[1]))
                    insulin_vals.append(float(parts[2]))
            
            if len(glucose_vals) >= 4:
                g0, g30, g60, g90 = glucose_vals[:4]
                i0, i30, i60, i90 = insulin_vals[:4]
                times = times[:4]
            else:
                st.error("Please provide at least 4 time points")
                g0, g30, g60, g90 = 88, 119, 92, 80
                i0, i30, i60, i90 = 8, 50, 70, 40
                times = [0, 30, 60, 90]
        except:
            st.error("Error parsing data. Please check format.")
            g0, g30, g60, g90 = 88, 119, 92, 80
            i0, i30, i60, i90 = 8, 50, 70, 40
            times = [0, 30, 60, 90]
    else:
        # Default values
        g0, g30, g60, g90 = 88, 119, 92, 80
        i0, i30, i60, i90 = 8, 50, 70, 40
        times = [0, 30, 60, 90]

# Data preparation
glucose = [g0, g30, g60, g90]
insulin = [i0, i30, i60, i90]

# Reference patterns (simplified)
normal_glucose = [90, 140, 120, 95]
normal_insulin = [6, 40, 30, 20]

# IMPROVED CLASSIFICATION ALGORITHMS
def calculate_glucotype(glucose_vals):
    """
    Improved glucotype classification based on Hall et al. research
    Focus on proper monophasic vs biphasic identification
    """
    g0, g30, g60, g90 = glucose_vals
    
    # Calculate incremental changes
    delta_30 = g30 - g0
    delta_60 = g60 - g30
    delta_90 = g90 - g60
    
    # Additional metrics
    peak_glucose = max(glucose_vals)
    peak_time = glucose_vals.index(peak_glucose) * 30
    
    # Return to baseline check (within 10 mg/dL of fasting)
    returns_to_baseline = g90 <= (g0 + 10)
    
    # Classify based on response patterns
    if delta_30 <= 0:
        return "Flat/Low Responder", "concern"
    elif peak_glucose < 100:
        return "Flat/Low Responder", "concern"
    elif delta_30 > 0 and delta_60 > 0 and delta_90 > 0:
        return "Continuous Rise", "concern"
    elif peak_time == 30:
        # Peak at 30min - check what happens after
        if delta_60 <= -20 and delta_90 > 10:
            # Sharp drop then rise = biphasic
            return "Biphasic", "normal"
        elif delta_60 <= 0 and returns_to_baseline:
            # Drops and stays down = good monophasic
            return "Monophasic", "warning"
        elif delta_60 > 0:
            # Continues rising after 30min peak = problem
            return "Continuous Rise", "concern"
        else:
            return "Monophasic", "warning"
    elif peak_time == 60:
        # Peak at 60min
        if delta_90 <= -15:
            # Good decline after 60min peak
            if returns_to_baseline:
                return "Delayed Monophasic", "warning"
            else:
                return "Delayed Peak", "warning"
        else:
            return "Sustained Elevation", "concern"
    elif peak_time == 90:
        return "Late Peak/Continuous Rise", "concern"
    else:
        # Check for true biphasic pattern
        if g30 > g0 and g60 < g30 and g90 > g60 and g90 < g30:
            return "Biphasic", "normal"
        else:
            return "Atypical Pattern", "warning"

def calculate_kraft_type(insulin_vals, glucose_vals=None):
    """
    Improved Kraft classification with glucose context
    Considers glucose disposal efficiency to avoid false Type V diagnosis
    """
    i0, i30, i60, i90 = insulin_vals
    
    # Calculate glucose disposal efficiency if available
    glucose_disposal_efficient = False
    if glucose_vals:
        g0, g30, g60, g90 = glucose_vals
        # Good glucose disposal: returns close to baseline despite lower insulin
        glucose_disposal_efficient = (g90 <= g0 + 15) and (max(glucose_vals) < 140)
    
    # Type IV: High fasting insulin
    if i0 >= 25:
        return "Type IV - High Fasting", "concern"
    
    # Type V: Low insulin response - BUT check glucose disposal first
    peak_insulin = max(insulin_vals)
    if peak_insulin < 30:
        if glucose_disposal_efficient:
            # Low insulin but good glucose disposal = high insulin sensitivity
            return "Type I - High Sensitivity", "normal"
        else:
            return "Type V - Low Response", "concern"
    
    # Find peak time
    peak_time_index = insulin_vals.index(peak_insulin)
    peak_time = peak_time_index * 30
    
    # More nuanced Type I criteria
    if peak_time == 30:
        if i90 < (i0 + 15) and i90 < 60:
            # Good return toward baseline
            if i90 < 30:
                return "Type I - Normal", "normal"
            else:
                return "Type I/II - Borderline", "warning"
        elif i90 >= 60:
            return "Type III - Sustained", "warning"
        else:
            return "Type II - Early Pattern", "warning"
    
    # Type II: Delayed peak
    elif peak_time >= 60:
        if i90 < 50:
            return "Type II - Delayed Peak", "warning"
        else:
            return "Type II/III - Delayed Sustained", "concern"
    
    # Type III: Sustained elevation
    elif i90 >= 60 or i90 > (i0 + 30):
        return "Type III - Sustained", "warning"
    
    # Default classification
    else:
        if i90 < 40:
            return "Type I - Normal", "normal"
        else:
            return "Type II - Early Pattern", "warning"

def calculate_metrics(glucose_vals, insulin_vals):
    """Enhanced metabolic metrics with additional calculations"""
    g0, g30, g60, g90 = glucose_vals
    i0, i30, i60, i90 = insulin_vals
    
    # HOMA-IR (standard)
    homa_ir = (g0 * i0) / 405
    
    # Matsuda Index (whole body insulin sensitivity)
    mean_glucose = np.mean(glucose_vals)
    mean_insulin = np.mean(insulin_vals)
    matsuda = 10000 / math.sqrt(g0 * i0 * mean_glucose * mean_insulin)
    
    # Area under curve using trapezoidal rule
    glucose_auc = 0.5 * (g0 + g30) * 30 + 0.5 * (g30 + g60) * 30 + 0.5 * (g60 + g90) * 30
    insulin_auc = 0.5 * (i0 + i30) * 30 + 0.5 * (i30 + i60) * 30 + 0.5 * (i60 + i90) * 30
    
    # Additional metrics
    glucose_peak = max(glucose_vals)
    insulin_peak = max(insulin_vals)
    
    # Insulin sensitivity indices
    isi_composite = matsuda  # Alternative name
    
    # Disposition index (beta cell function adjusted for insulin sensitivity)
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
        'disposition_index': disposition_index,
        'isi_composite': isi_composite
    }

# Enhanced plotting with CORRECTED shading logic
def create_clean_plot(x_data, y_data, title, y_label, reference_data, reference_name, color, is_glucose=True):
    fig = go.Figure()
    
    # Add conditional shading FIRST (behind everything) if enabled
    if show_shading:
        for i in range(len(x_data) - 1):
            # Get the points for this segment
            x1, x2 = x_data[i], x_data[i+1]
            y1_user, y2_user = y_data[i], y_data[i+1]
            y1_ref, y2_ref = reference_data[i], reference_data[i+1]
            
            # Determine the area between the lines
            # We need to check which line is on top for each segment
            user_above_start = y1_user > y1_ref
            user_above_end = y2_user > y2_ref
            
            if user_above_start and user_above_end:
                # User completely above reference - BAD (RED)
                fill_color = 'rgba(231, 76, 60, 0.05)'  # Light red
                x_fill = [x1, x2, x2, x1]
                y_fill = [y1_ref, y2_ref, y2_user, y1_user]
            elif not user_above_start and not user_above_end:
                # User completely below reference - GOOD (GREEN)
                fill_color = 'rgba(46, 204, 113, 0.03)'  # Light green
                x_fill = [x1, x2, x2, x1]
                y_fill = [y1_user, y2_user, y2_ref, y1_ref]
            else:
                # Lines cross - need to handle intersection
                # For simplicity, use average to determine predominant relationship
                user_avg = (y1_user + y2_user) / 2
                ref_avg = (y1_ref + y2_ref) / 2
                
                if user_avg > ref_avg:
                    fill_color = 'rgba(231, 76, 60, 0.15)'  # Light red
                    x_fill = [x1, x2, x2, x1]
                    y_fill = [y1_ref, y2_ref, y2_user, y1_user]
                else:
                    fill_color = 'rgba(46, 204, 113, 0.15)'  # Light green
                    x_fill = [x1, x2, x2, x1]
                    y_fill = [y1_user, y2_user, y2_ref, y1_ref]
            
            # Add fill area for this segment
            fig.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='toself',
                fillcolor=fill_color,
                line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Reference line (subtle) - only show if enabled
    if show_reference:
        fig.add_trace(go.Scatter(
            x=x_data, 
            y=reference_data,
            mode='lines',
            name=f'{reference_name} Pattern',
            line=dict(color=reference_color, width=reference_width, dash='dot'),
            opacity=reference_opacity
        ))
        
        # Reference annotations if enabled
        if show_reference_annotations:
            for i, (x, y) in enumerate(zip(x_data, reference_data)):
                if annotation_bold:
                    ref_text_content = f"<b>{y:.0f}</b>"
                else:
                    ref_text_content = f"{y:.0f}"
                
                fig.add_annotation(
                    x=x, y=y,
                    text=ref_text_content,
                    showarrow=False,
                    yshift=20,
                    font=dict(
                        family="Avenir, sans-serif", 
                        size=annotation_size * 0.85,  # Slightly smaller than main annotations
                        color=reference_color
                    ),
                    bgcolor="rgba(255, 255, 255, 0)",
                    bordercolor="rgba(0, 0, 0, 0)",
                    borderwidth=0
                )
    
    # User data (prominent) - add this after shading so it appears on top
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=y_data,
        mode='lines+markers',
        name='Your Data',
        line=dict(color=color, width=line_width),
        marker=dict(size=marker_size, color=color, line=dict(color='white', width=2))
    ))
    
    # Data annotations - transparent background
    if show_annotations:
        yshift = 20
        
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            if annotation_bold:
                text_content = f"<b>{y:.0f}</b>"
            else:
                text_content = f"{y:.0f}"
            
            fig.add_annotation(
                x=x, y=y,
                text=text_content,
                showarrow=False,
                yshift=yshift,
                font=dict(
                    family="Avenir, sans-serif", 
                    size=annotation_size, 
                    color=annotation_color
                ),
                bgcolor="rgba(255, 255, 255, 0)",  # Fully transparent background
                bordercolor="rgba(0, 0, 0, 0)",  # Transparent border
                borderwidth=0
            )
    
    # Typography-focused layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Cormorant Garamond, serif", size=title_font_size, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        xaxis=dict(
            title=dict(
                text="Time (minutes)",
                font=dict(family="Cormorant Garamond, serif", size=axis_title_size, color='#34495e')
            ),
            tickfont=dict(family="Avenir, sans-serif", size=axis_tick_size, color='#7f8c8d'),
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1,
            tickmode='array',
            tickvals=x_data,
            ticktext=[f"{int(x)}" for x in x_data]
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(family="Cormorant Garamond, serif", size=axis_title_size, color='#34495e')
            ),
            tickfont=dict(family="Avenir, sans-serif", size=axis_tick_size, color='#7f8c8d'),
            showgrid=True,
            gridcolor='#ecf0f1',
            gridwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(t=60, b=40, l=60, r=40),
        font=dict(family="Avenir, sans-serif")
    )
    
    return fig

# Create plots (removed dedicated PNG download buttons)
st.markdown("### Glucose Response")
fig1 = create_clean_plot(
    times, glucose, 
    'Glucose Response',
    'Glucose (mg/dL)',
    normal_glucose,
    'Normal',
    glucose_color,
    is_glucose=True
)

# Add download config to the chart
st.plotly_chart(fig1, use_container_width=True, config={
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['downloadImage'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'glucose_response',
        'height': download_height,
        'width': download_width,
        'scale': download_scale
    }
})

st.markdown("### Insulin Response")
fig2 = create_clean_plot(
    times, insulin,
    'Insulin Response', 
    'Insulin (μU/mL)',
    normal_insulin,
    'Normal',
    insulin_color,
    is_glucose=False
)

st.plotly_chart(fig2, use_container_width=True, config={
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['downloadImage'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'insulin_response',
        'height': download_height,
        'width': download_width,
        'scale': download_scale
    }
})

# Calculate results with improved algorithms
glucotype, gluco_status = calculate_glucotype(glucose)
kraft_type, kraft_status = calculate_kraft_type(insulin)
metrics = calculate_metrics(glucose, insulin)

# Determine HOMA-IR status with better thresholds
if metrics['homa_ir'] < 1.0:
    homa_status = "normal"
elif metrics['homa_ir'] < 2.5:
    homa_status = "warning"
else:
    homa_status = "concern"

# Clean results display
st.markdown("### Analysis Results")

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

# Enhanced metrics display
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Matsuda Index", f"{metrics['matsuda']:.1f}", help="Insulin sensitivity (higher = better)")
    st.metric("Glucose AUC", f"{metrics['glucose_auc']:.0f}", help="Total glucose exposure")

with col2:
    st.metric("Peak Glucose", f"{metrics['glucose_peak']:.0f} mg/dL", help="Highest glucose level")
    st.metric("Peak Insulin", f"{metrics['insulin_peak']:.0f} μU/mL", help="Highest insulin level")

with col3:
    st.metric("Insulinogenic Index", f"{metrics['insulinogenic_index']:.2f}", help="Early insulin secretion")
    st.metric("Disposition Index", f"{metrics['disposition_index']:.2f}", help="Beta cell function")

# Enhanced interpretation
st.markdown("### Interpretation")

interpretations = []

# Glucose interpretation
if gluco_status == "normal":
    interpretations.append("✓ **Glucose handling**: Healthy biphasic response with proper peak and return to baseline")
elif gluco_status == "warning":
    interpretations.append("⚠ **Glucose handling**: Intermediate response pattern - may indicate early metabolic stress")
else:
    interpretations.append("⚠ **Glucose handling**: Concerning pattern - suggests impaired glucose metabolism")

# Kraft interpretation with more detail
if kraft_status == "normal":
    interpretations.append("✓ **Insulin response**: Normal Kraft Type I pattern - healthy insulin dynamics")
elif kraft_status == "warning":
    if "Type II" in kraft_type:
        interpretations.append("⚠ **Insulin response**: Kraft Type II - delayed insulin peak, early insulin resistance")
    elif "Type III" in kraft_type:
        interpretations.append("⚠ **Insulin response**: Kraft Type III - sustained insulin elevation, progressing insulin resistance")
    else:
        interpretations.append("⚠ **Insulin response**: Early signs of insulin dysfunction")
else:
    if "Type IV" in kraft_type:
        interpretations.append("⚠ **Insulin response**: Kraft Type IV - high fasting insulin, established insulin resistance")
    elif "Type V" in kraft_type:
        interpretations.append("⚠ **Insulin response**: Kraft Type V - low insulin response, possible beta cell dysfunction")
    else:
        interpretations.append("⚠ **Insulin response**: Significant insulin dysfunction detected")

# HOMA-IR interpretation
if homa_status == "normal":
    interpretations.append("✓ **Insulin sensitivity**: Normal HOMA-IR, good insulin sensitivity")
elif homa_status == "warning":
    interpretations.append("⚠ **Insulin sensitivity**: Borderline HOMA-IR (1.0-2.5), mild insulin resistance")
else:
    interpretations.append("⚠ **Insulin sensitivity**: Elevated HOMA-IR (>2.5), significant insulin resistance")

# Additional interpretations based on new metrics
if metrics['matsuda'] > 4.0:
    interpretations.append("✓ **Whole body insulin sensitivity**: Excellent Matsuda Index (>4.0)")
elif metrics['matsuda'] > 2.5:
    interpretations.append("⚠ **Whole body insulin sensitivity**: Moderate Matsuda Index (2.5-4.0)")
else:
    interpretations.append("⚠ **Whole body insulin sensitivity**: Low Matsuda Index (<2.5), insulin resistance")

if metrics['disposition_index'] > 1.0:
    interpretations.append("✓ **Beta cell function**: Good disposition index, adequate insulin secretion capacity")
else:
    interpretations.append("⚠ **Beta cell function**: Low disposition index, may indicate beta cell stress")

for interpretation in interpretations:
    st.markdown(interpretation)

# Debug information (optional - can be hidden in sidebar)
with st.sidebar:
    st.markdown("---")
    if st.checkbox("Show Debug Info", value=False):
        st.subheader("Debug Information")
        st.write("**Raw Values:**")
        st.write(f"Glucose: {glucose}")
        st.write(f"Insulin: {insulin}")
        st.write(f"**Glucose Deltas:**")
        st.write(f"0-30min: {glucose[1] - glucose[0]:.1f}")
        st.write(f"30-60min: {glucose[2] - glucose[1]:.1f}")
        st.write(f"60-90min: {glucose[3] - glucose[2]:.1f}")
        st.write(f"**Insulin Analysis:**")
        st.write(f"Peak: {max(insulin):.1f} at {insulin.index(max(insulin)) * 30}min")
        st.write(f"90min level: {insulin[3]:.1f}")

# Simple export with enhanced data
st.markdown("### Export Data")
export_data = {
    'Date': datetime.now().strftime('%Y-%m-%d'),
    'Glucotype': glucotype,
    'Glucotype_Status': gluco_status,
    'Kraft_Type': kraft_type,
    'Kraft_Status': kraft_status,
    'HOMA_IR': round(metrics['homa_ir'], 2),
    'HOMA_IR_Status': homa_status,
    'Matsuda_Index': round(metrics['matsuda'], 1),
    'Disposition_Index': round(metrics['disposition_index'], 2),
    'Insulinogenic_Index': round(metrics['insulinogenic_index'], 2),
    'Glucose_AUC': round(metrics['glucose_auc'], 0),
    'Insulin_AUC': round(metrics['insulin_auc'], 0),
    'Glucose_Peak': round(metrics['glucose_peak'], 0),
    'Insulin_Peak': round(metrics['insulin_peak'], 0),
    'Glucose_0': g0, 'Glucose_30': g30, 'Glucose_60': g60, 'Glucose_90': g90,
    'Insulin_0': i0, 'Insulin_30': i30, 'Insulin_60': i60, 'Insulin_90': i90
}

df = pd.DataFrame([export_data])
st.download_button(
    "Download Results (CSV)",
    df.to_csv(index=False),
    f"ogtt_results_{datetime.now().strftime('%Y%m%d')}.csv",
    "text/csv"
)

# Simple disclaimer
st.markdown("---")
st.markdown("*This tool is for educational purposes. Consult healthcare professionals for medical advice.*")