#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Glucose Test Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .interpretation-box {
        background: #f0f7ff;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>Glucose Test Analyzer</h1><p>Upload your test data to see results and trends</p></div>', unsafe_allow_html=True)

# File upload section
st.subheader("Step 1: Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file with your glucose and insulin test results",
    type=['csv'],
    help="Your file should have columns: time, type, [test dates], reference"
)

# Load sample data if no file uploaded
if uploaded_file is None:
    st.info("No file uploaded. Showing sample data below.")
    # Simple sample data
    sample_data = {
        'time': [0, 30, 60, 90, 120] * 2,
        'type': ['glucose'] * 5 + ['insulin'] * 5,
        'March_2024': [95, 165, 145, 120, 100, 8, 45, 60, 40, 15],
        'June_2024': [88, 140, 125, 105, 90, 6, 35, 50, 30, 12],
        'reference': [90, 140, 130, 110, 100, 5, 30, 40, 25, 10]
    }
    df = pd.DataFrame(sample_data)
else:
    df = pd.read_csv(uploaded_file)

# Process data
def get_test_dates(df):
    return [col for col in df.columns if col not in ['time', 'type', 'reference']]

def split_data(df):
    glucose_data = df[df['type'] == 'glucose'].reset_index(drop=True)
    insulin_data = df[df['type'] == 'insulin'].reset_index(drop=True)
    return glucose_data, insulin_data

glucose_df, insulin_df = split_data(df)
test_dates = get_test_dates(df)

if len(test_dates) == 0:
    st.error("No test data found. Please check your file format.")
    st.stop()

# Step 2: Select test to analyze
st.subheader("Step 2: Choose Test to Analyze")
selected_test = st.selectbox("Select a test date:", test_dates)

# Step 3: Create visualization
st.subheader("Step 3: Your Results")

# Create simple, clean chart
fig = go.Figure()

# Get data for selected test
time_points = glucose_df['time'].values
glucose_values = glucose_df[selected_test].values
insulin_values = insulin_df[selected_test].values
glucose_reference = glucose_df['reference'].values
insulin_reference = insulin_df['reference'].values

# Add glucose traces
fig.add_trace(go.Scatter(
    x=time_points,
    y=glucose_values,
    mode='lines+markers',
    name='Your Glucose',
    line=dict(color='#e74c3c', width=3),
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=time_points,
    y=glucose_reference,
    mode='lines',
    name='Normal Range',
    line=dict(color='#95a5a6', width=2, dash='dash'),
    opacity=0.7
))

# Update layout
fig.update_layout(
    title=f'Glucose Response - {selected_test.replace("_", " ")}',
    xaxis_title='Time (minutes)',
    yaxis_title='Glucose (mg/dL)',
    template='plotly_white',
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

# Create insulin chart
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=time_points,
    y=insulin_values,
    mode='lines+markers',
    name='Your Insulin',
    line=dict(color='#3498db', width=3),
    marker=dict(size=8)
))

fig2.add_trace(go.Scatter(
    x=time_points,
    y=insulin_reference,
    mode='lines',
    name='Normal Range',
    line=dict(color='#95a5a6', width=2, dash='dash'),
    opacity=0.7
))

fig2.update_layout(
    title=f'Insulin Response - {selected_test.replace("_", " ")}',
    xaxis_title='Time (minutes)',
    yaxis_title='Insulin (µU/mL)',
    template='plotly_white',
    height=400,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig2, use_container_width=True)

# Step 4: Simple analysis
st.subheader("Step 4: What Your Results Mean")

# Calculate basic metrics
fasting_glucose = glucose_values[0]
peak_glucose = max(glucose_values)
fasting_insulin = insulin_values[0]
peak_insulin = max(insulin_values)

# Show key numbers
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Fasting Glucose", f"{fasting_glucose:.0f} mg/dL")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Peak Glucose", f"{peak_glucose:.0f} mg/dL")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Fasting Insulin", f"{fasting_insulin:.0f} µU/mL")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Peak Insulin", f"{peak_insulin:.0f} µU/mL")
    st.markdown('</div>', unsafe_allow_html=True)

# Simple interpretation
interpretation = []

# Glucose interpretation
if fasting_glucose >= 126:
    interpretation.append("**Fasting Glucose**: High - may indicate diabetes")
elif fasting_glucose >= 100:
    interpretation.append("**Fasting Glucose**: Elevated - prediabetic range")
else:
    interpretation.append("**Fasting Glucose**: Normal")

if len(glucose_values) >= 3:  # If we have 2-hour data
    two_hour_glucose = glucose_values[-1]
    if two_hour_glucose >= 200:
        interpretation.append("**2-Hour Glucose**: High - may indicate diabetes")
    elif two_hour_glucose >= 140:
        interpretation.append("**2-Hour Glucose**: Elevated - impaired glucose tolerance")
    else:
        interpretation.append("**2-Hour Glucose**: Normal")

# Insulin interpretation
if fasting_insulin > 25:
    interpretation.append("**Fasting Insulin**: High - may indicate insulin resistance")
elif fasting_insulin < 2:
    interpretation.append("**Fasting Insulin**: Low - may indicate reduced insulin production")
else:
    interpretation.append("**Fasting Insulin**: Normal")

# Display interpretation
st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
st.markdown("### Your Results Summary")
for item in interpretation:
    st.markdown(f"• {item}")

st.markdown("---")
st.markdown("**Note**: These results are for informational purposes only. Always consult with your healthcare provider for medical advice.")
st.markdown('</div>', unsafe_allow_html=True)

# Optional: Compare multiple tests
if len(test_dates) > 1:
    st.subheader("Compare Your Tests")
    
    comparison_dates = st.multiselect(
        "Select tests to compare:",
        test_dates,
        default=test_dates[:2] if len(test_dates) >= 2 else test_dates
    )
    
    if len(comparison_dates) > 1:
        # Create comparison chart
        fig_comp = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, date in enumerate(comparison_dates):
            glucose_vals = glucose_df[date].values
            fig_comp.add_trace(go.Scatter(
                x=time_points,
                y=glucose_vals,
                mode='lines+markers',
                name=date.replace('_', ' '),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))
        
        fig_comp.update_layout(
            title='Glucose Comparison Across Tests',
            xaxis_title='Time (minutes)',
            yaxis_title='Glucose (mg/dL)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

# Data editing section
with st.expander("Edit Your Data"):
    st.markdown("You can edit your data directly in the table below:")
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    
    if st.button("Update Charts with Edited Data"):
        # Update the dataframes
        glucose_df, insulin_df = split_data(edited_df)
        st.success("Data updated! Scroll up to see the new charts.")
        st.rerun()

# Download section
st.subheader("Download Your Results")
col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"glucose_test_{selected_test}.csv",
        mime="text/csv"
    )

with col2:
    # Create a simple summary report
    report = f"""
# Glucose Test Report - {selected_test.replace('_', ' ')}

## Key Results
- Fasting Glucose: {fasting_glucose:.0f} mg/dL
- Peak Glucose: {peak_glucose:.0f} mg/dL  
- Fasting Insulin: {fasting_insulin:.0f} µU/mL
- Peak Insulin: {peak_insulin:.0f} µU/mL

## Interpretation
{chr(10).join(f"• {item}" for item in interpretation)}

*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""
    
    st.download_button(
        label="Download Summary Report",
        data=report,
        file_name=f"glucose_report_{selected_test}.txt",
        mime="text/plain"
    )

# Help section
with st.expander("Need Help?"):
    st.markdown("""
    **How to use this tool:**
    1. Upload a CSV file with your test data
    2. Select which test date you want to analyze
    3. View your glucose and insulin charts
    4. Read the interpretation of your results
    
    **Data format:**
    Your CSV should have these columns:
    - `time`: Time points (0, 30, 60, 90, 120 minutes)
    - `type`: Either 'glucose' or 'insulin'
    - `[your_test_date]`: Your actual test values
    - `reference`: Normal reference values
    
    **Normal ranges:**
    - Fasting glucose: 70-99 mg/dL
    - 2-hour glucose: <140 mg/dL
    - Fasting insulin: 2-25 µU/mL
    """)