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

# Sidebar controls
with st.sidebar:
    st.header("Display Controls")
    
    # Shading toggle
    show_shading = st.checkbox("Show comparative shading", value=True)
    
    # Line appearance controls
    st.subheader("Line Appearance")
    glucose_color = st.color_picker("Glucose line color", "#2E86C1")  # Nice blue
    insulin_color = st.color_picker("Insulin line color", "#3498DB")  # Slightly lighter blue
    
    # Reference line controls
    show_reference = st.checkbox("Show reference line", value=True)
    reference_color = st.color_picker("Reference line color", "#1cb164")
    reference_opacity = st.slider("Reference line opacity", 0.1, 1.0, 0.5, 0.1)
    reference_width = st.slider("Reference line width", 1, 10, 3)  # Increased default
    
    line_width = st.slider("Line width", 1, 10, 5)  # Increased default
    marker_size = st.slider("Marker size", 4, 20, 13)  # Increased default
    
    # Annotation controls
    st.subheader("Annotation Settings")
    show_annotations = st.checkbox("Show value annotations", value=True)
    show_reference_annotations = st.checkbox("Show reference point annotations", value=False)
    annotation_size = st.slider("Annotation text size", 8, 48, 24)  # Increased default
    annotation_position = st.selectbox("Annotation position", ["Above", "Below"], index=0)
    annotation_bold = st.checkbox("Bold annotations", value=True)
    annotation_color = st.color_picker("Annotation color", "#2c3e50")
    
    # Font size controls
    st.subheader("Chart Font Sizes")
    title_font_size = st.slider("Chart title size", 12, 72, 38)  # Increased default
    axis_title_size = st.slider("Axis title size", 8, 48, 24)  # Increased default
    axis_tick_size = st.slider("Axis tick size", 6, 36, 18)  # Increased default
    
    # Download resolution controls
    st.subheader("Download Settings")
    download_width = st.slider("Download width (px)", 800, 4000, 1200, step=200)  # PowerPoint optimized
    download_height = st.slider("Download height (px)", 400, 3000, 550, step=50)  # 16:9 aspect ratio
    download_scale = st.slider("Download scale multiplier", 1, 10, 1)  # Lighter default for presentations
    
    st.write(f"Final resolution: {download_width * download_scale} × {download_height * download_scale} px")
    st.caption("💡 Default settings optimized for PowerPoint/Keynote presentations")
    st.subheader("Data Input Method")
    input_method = st.radio("Choose input method:", ["Manual Entry", "CSV Upload", "Paste Data"])

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

# Core calculations
def calculate_glucotype(glucose_vals):
    """Simplified glucotype classification"""
    d1 = glucose_vals[1] - glucose_vals[0]  # 0-30
    d2 = glucose_vals[2] - glucose_vals[1]  # 30-60
    d3 = glucose_vals[3] - glucose_vals[2]  # 60-90
    
    if d1 <= 0:
        return "Abnormal", "concern"
    elif d2 > 0 and d3 > 0:
        return "Incessant Rise", "concern"
    elif d2 > 0 and d3 < -4.5:
        return "Monophasic", "warning"
    elif d2 < -4.5 and d3 > 4.5:
        return "Biphasic", "normal"
    else:
        return "Atypical", "warning"

def calculate_kraft_type(insulin_vals):
    """Simplified Kraft classification"""
    fasting = insulin_vals[0]
    peak = max(insulin_vals[1:])
    peak_time = (insulin_vals[1:].index(peak) + 1) * 30
    
    if fasting > 25:
        return "Type IV (High Fasting)", "concern"
    elif peak < 20:
        return "Type V (Low Response)", "concern"
    elif peak_time == 30 and insulin_vals[3] < 50:
        return "Type I (Normal)", "normal"
    elif peak_time == 60:
        return "Type II (Delayed)", "warning"
    else:
        return "Type III (Sustained)", "warning"

def calculate_metrics(glucose_vals, insulin_vals):
    """Core metabolic metrics"""
    homa_ir = (glucose_vals[0] * insulin_vals[0]) / 405
    
    mean_g = np.mean(glucose_vals)
    mean_i = np.mean(insulin_vals)
    matsuda = 10000 / math.sqrt(glucose_vals[0] * insulin_vals[0] * mean_g * mean_i)
    
    # Area under curve (simplified trapezoidal)
    glucose_auc = sum((glucose_vals[i] + glucose_vals[i+1]) * 15 for i in range(3))  # 30min intervals
    insulin_auc = sum((insulin_vals[i] + insulin_vals[i+1]) * 15 for i in range(3))
    
    return {
        'homa_ir': homa_ir,
        'matsuda': matsuda,
        'glucose_auc': glucose_auc,
        'insulin_auc': insulin_auc
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
                    yshift=20, #if annotation_position == "Above" else 20,  # Opposite of main annotations
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
        yshift = 20 # if annotation_position == "Above" else -15
        
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

# Create plots with download buttons
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("### Glucose Response")
with col2:
    if st.button("📥 PNG", key="glucose_png", help="Download high-quality PNG"):
        # This will be handled by browser download
        pass

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

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("### Insulin Response")
with col2:
    if st.button("📥 PNG", key="insulin_png", help="Download high-quality PNG"):
        # This will be handled by browser download
        pass

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

# Calculate results
glucotype, gluco_status = calculate_glucotype(glucose)
kraft_type, kraft_status = calculate_kraft_type(insulin)
metrics = calculate_metrics(glucose, insulin)

# Determine HOMA-IR status
if metrics['homa_ir'] < 1.0:
    homa_status = "normal"
elif metrics['homa_ir'] > 2.5:
    homa_status = "concern"
else:
    homa_status = "warning"

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

# Key metrics table
st.markdown("### Key Metrics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Matsuda Index", f"{metrics['matsuda']:.1f}", help="Insulin sensitivity (higher = better)")
    st.metric("Glucose AUC", f"{metrics['glucose_auc']:.0f}", help="Total glucose exposure")

with col2:
    st.metric("Peak Glucose", f"{max(glucose):.0f} mg/dL", help="Highest glucose level")
    st.metric("Peak Insulin", f"{max(insulin):.0f} μU/mL", help="Highest insulin level")

# Simple interpretation
st.markdown("### Interpretation")

interpretations = []

if gluco_status == "normal":
    interpretations.append("✓ **Glucose handling**: Optimal biphasic response pattern")
elif gluco_status == "warning":
    interpretations.append("⚠ **Glucose handling**: Intermediate response pattern")
else:
    interpretations.append("⚠ **Glucose handling**: Concerning response pattern")

if kraft_status == "normal":
    interpretations.append("✓ **Insulin response**: Normal pattern")
elif kraft_status == "warning":
    interpretations.append("⚠ **Insulin response**: Early insulin resistance signs")
else:
    interpretations.append("⚠ **Insulin response**: Significant insulin dysfunction")

if homa_status == "normal":
    interpretations.append("✓ **Insulin sensitivity**: Normal range")
elif homa_status == "warning":
    interpretations.append("⚠ **Insulin sensitivity**: Borderline insulin resistance")
else:
    interpretations.append("⚠ **Insulin sensitivity**: Insulin resistance present")

for interpretation in interpretations:
    st.markdown(interpretation)

# Simple export
st.markdown("### Export Data")
export_data = {
    'Date': datetime.now().strftime('%Y-%m-%d'),
    'Glucotype': glucotype,
    'Kraft_Type': kraft_type,
    'HOMA_IR': round(metrics['homa_ir'], 2),
    'Matsuda_Index': round(metrics['matsuda'], 1),
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