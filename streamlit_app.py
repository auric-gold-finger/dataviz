import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="OGTT Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- (Optional) Check for Kaleido for PNG export ---
try:
    import kaleido
    KALEIDO_INSTALLED = True
except ImportError:
    KALEIDO_INSTALLED = False

# --- Helper Functions for Calculations ---

def calculate_matsuda(df):
    """Calculates the Matsuda Index."""
    try:
        g0 = df.loc[df['Time'] == 0, 'Glucose'].iloc[0]
        i0 = df.loc[df['Time'] == 0, 'Insulin'].iloc[0]
        mean_glucose = df['Glucose'].mean()
        mean_insulin = df['Insulin'].mean()

        if g0 <= 0 or i0 <= 0 or mean_glucose <= 0 or mean_insulin <= 0:
            return np.nan

        matsuda_index = 10000 / np.sqrt((g0 * i0) * (mean_glucose * mean_insulin))
        return matsuda_index
    except (IndexError, ZeroDivisionError):
        return np.nan

def calculate_homa_ir(df):
    """Calculates HOMA-IR."""
    try:
        g0 = df.loc[df['Time'] == 0, 'Glucose'].iloc[0]
        i0 = df.loc[df['Time'] == 0, 'Insulin'].iloc[0]
        homa_ir = (g0 * i0) / 405
        return homa_ir
    except (IndexError, ZeroDivisionError):
        return np.nan

def calculate_insulinogenic_index(df):
    """Calculates the Insulinogenic Index (IGI)."""
    try:
        g0 = df.loc[df['Time'] == 0, 'Glucose'].iloc[0]
        i0 = df.loc[df['Time'] == 0, 'Insulin'].iloc[0]
        g30 = df.loc[df['Time'] == 30, 'Glucose'].iloc[0]
        i30 = df.loc[df['Time'] == 30, 'Insulin'].iloc[0]

        if (g30 - g0) == 0:
            return np.nan # Avoid division by zero
        igi = (i30 - i0) / (g30 - g0)
        return igi
    except IndexError:
        return np.nan


def determine_kraft_type(df):
    """Determines Kraft Type based on insulin levels at specific time points."""
    try:
        i0 = df.loc[df['Time'] == 0, 'Insulin'].iloc[0]
        i30 = df.loc[df['Time'] == 30, 'Insulin'].iloc[0]
        i60 = df.loc[df['Time'] == 60, 'Insulin'].iloc[0]
        i120 = df.loc[df['Time'] == 120, 'Insulin'].iloc[0]
        peak_insulin = df['Insulin'].max()
        peak_time = df.loc[df['Insulin'].idxmax(), 'Time']

        # The order of these checks is important
        # Pattern I: Normal
        if i120 < i0 and peak_insulin < 100 and peak_time <= 60:
             return "Pattern I: Normal", "Normal insulin response. Insulin peaks early and returns to or below baseline by 2 hours."

        # Pattern V: Diabetic response with beta-cell failure (low and flat)
        if i0 > 30 and i30 < i0 and i60 < i0 and i120 < i0:
            return "Pattern V: Diabetic Response (Beta-cell Failure)", "Low, flat insulin curve, suggesting severe beta-cell exhaustion. Often seen in established Type 1 or late-stage Type 2 Diabetes."

        # Pattern IV: Irreversible IR / Diabetes (high fasting, stays high)
        if i0 >= 30 and i120 >= 50:
             return "Pattern IV: Irreversible Insulin Resistance", "High fasting insulin and a sustained high insulin level at 2 hours, indicating severe and established insulin resistance, consistent with Type 2 Diabetes."

        # Pattern III: Delayed Peak Hyperinsulinemia
        if peak_time > 60 and i120 > i0:
            if i0 < 30:
                 return "Pattern IIIa: Hyperinsulinemia (Delayed Peak)", "Insulin response is delayed, peaking late (at or after 2 hours) and remaining high. This is a clear sign of insulin resistance."
            else:
                 return "Pattern IIIb: Hyperinsulinemia (High Fasting & Delayed Peak)", "High fasting insulin combined with a delayed peak indicates significant, advanced insulin resistance."

        # Pattern II: Compensatory Hyperinsulinemia
        if i0 < 30 and i120 > i0:
             return "Pattern II: Hyperinsulinemia (Compensatory)", "Fasting insulin is normal, but the response to glucose is exaggerated and prolonged. This is a classic sign of the body compensating for insulin resistance."

        return "Unclassified", "The insulin pattern does not fit a standard Kraft type. This may be due to unusual metabolic responses or data entry issues."

    except IndexError:
        return "Incomplete Data", "Cannot determine Kraft Type because data for required time points (0, 30, 60, 120 min) is missing."


def determine_glucotype(df):
    """Determines Glucotype based on glucose curve shape."""
    try:
        g0 = df.loc[df['Time'] == 0, 'Glucose'].iloc[0]
        g120 = df.loc[df['Time'] == 120, 'Glucose'].iloc[0]
        peak_glucose = df['Glucose'].max()
        peak_time = df.loc[df['Glucose'].idxmax(), 'Time']

        if g120 >= 140:
            return "Severe Variability (Impaired Glucose Tolerance)", "Glucose remains high (≥140 mg/dL) at the 2-hour mark, meeting the criteria for Impaired Glucose Tolerance or potentially Diabetes, depending on the value."
        elif peak_glucose < 140 and g120 < g0:
            return "Low Variability", "Excellent glucose control. Glucose levels remain very stable with minimal increase and return below fasting by 2 hours."
        elif peak_time <= 60 and peak_glucose >= 140:
             return "Moderate Variability (Classic Spike)", "A moderate, early glucose spike that resolves well below the 2-hour threshold for pre-diabetes. A common pattern."
        elif peak_time > 60 and peak_glucose >= 140:
            return "High Variability (Delayed Spike)", "Glucose peaks late in the test, indicating a delayed ability to handle the glucose load. This pattern is associated with a higher risk of impaired glucose tolerance."

        return "Unclassified", "The glucose pattern is unusual and does not fit a standard glucotype."
    except IndexError:
        return "Incomplete Data", "Cannot determine Glucotype due to missing data."


def create_ogtt_plot(df):
    """Creates a dual-axis Plotly chart for Glucose and Insulin from a single dataframe."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Add Glucose Trace ---
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Glucose'],
            name="Glucose (mg/dL)",
            mode='lines+markers+text',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8),
            text=df['Glucose'].round(0).astype(str),
            textposition='top center',
            textfont=dict(color='royalblue')
        ),
        secondary_y=False,
    )

    # --- Add Insulin Trace ---
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Insulin'],
            name="Insulin (µU/mL)",
            mode='lines+markers+text',
            line=dict(color='firebrick', width=3),
            marker=dict(size=8),
            text=df['Insulin'].round(1).astype(str),
            textposition='bottom center',
            textfont=dict(color='firebrick')
        ),
        secondary_y=True,
    )

    # --- Update Layout and Axes ---
    fig.update_layout(
        title_text="Oral Glucose Tolerance Test (OGTT) Results",
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    fig.update_xaxes(
        title_text="Time (minutes)",
        tickvals=df['Time'].unique() # Ensure all time points are shown
    )
    fig.update_yaxes(title_text="<b>Glucose</b> (mg/dL)", color='royalblue', secondary_y=False)
    fig.update_yaxes(title_text="<b>Insulin</b> (µU/mL)", color='firebrick', secondary_y=True)

    return fig

# --- Sidebar for Data Input ---
st.sidebar.header("Data Input")
st.sidebar.write("Paste your data in CSV format with columns: `Time`, `Glucose`, `Insulin`")

# Example data to guide the user
EXAMPLE_DATA = """Time,Glucose,Insulin
0,85,5.0
30,155,55.0
60,140,70.0
90,110,40.0
120,90,15.0
"""
st.sidebar.code(EXAMPLE_DATA, language='csv')

pasted_data = st.sidebar.text_area("Paste your CSV data here", height=200, placeholder=EXAMPLE_DATA)

df = None
if pasted_data:
    try:
        df = pd.read_csv(io.StringIO(pasted_data))
        # Basic validation
        if not all(col in df.columns for col in ['Time', 'Glucose', 'Insulin']):
            st.error("CSV must contain 'Time', 'Glucose', and 'Insulin' columns.")
            df = None
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        df = None

# --- Main Panel for Results ---
st.title("📈 OGTT Analysis Dashboard")
st.markdown("This tool analyzes Oral Glucose Tolerance Test (OGTT) data to provide insights into metabolic health, including insulin sensitivity and secretion.")

if df is not None and not df.empty and df.isnull().sum().sum() == 0:
    st.header("Analysis Results")

    # --- Data Table and Plot ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Data")
        st.dataframe(df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Data Visualization")
        fig = create_ogtt_plot(df)
        st.plotly_chart(fig, use_container_width=True)

        # Add download button for the plot
        if KALEIDO_INSTALLED:
            img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
            st.download_button(
                label="⬇️ Download Plot as PNG",
                data=img_bytes,
                file_name="ogtt_analysis_plot.png",
                mime="image/png"
            )
        else:
            st.info("To enable PNG download, install kaleido: `pip install kaleido`")


    st.divider()

    # --- Calculations ---
    matsuda = calculate_matsuda(df)
    homa_ir = calculate_homa_ir(df)
    igi = calculate_insulinogenic_index(df)
    kraft_type, kraft_desc = determine_kraft_type(df)
    glucotype, glucotype_desc = determine_glucotype(df)

    # --- Display Metrics ---
    st.subheader("Key Metabolic Indices")
    res_cols = st.columns(3)
    res_cols[0].metric(
        label="HOMA-IR",
        value=f"{homa_ir:.2f}" if not np.isnan(homa_ir) else "N/A",
        help="Estimates insulin resistance from fasting data. Lower is better (Optimal: < 1.0, Concern: > 1.9, High IR: > 2.9)."
    )
    res_cols[1].metric(
        label="Matsuda Index",
        value=f"{matsuda:.2f}" if not np.isnan(matsuda) else "N/A",
        help="Measures whole-body insulin sensitivity using all time points. Higher is better (Good: > 4.0)."
    )
    res_cols[2].metric(
        label="Insulinogenic Index (IGI₃₀)",
        value=f"{igi:.2f}" if not np.isnan(igi) else "N/A",
        help="Measures early-phase insulin secretion relative to glucose change. A lower value may indicate beta-cell dysfunction."
    )

    st.divider()

    # --- Display Classifications ---
    st.subheader("Metabolic Pattern Classification")
    class_cols = st.columns(2)
    with class_cols[0]:
        with st.container(border=True):
            st.markdown(f"#### Kraft Type: {kraft_type}")
            st.write(kraft_desc)

    with class_cols[1]:
        with st.container(border=True):
            st.markdown(f"#### Glucotype: {glucotype}")
            st.write(glucotype_desc)

else:
    st.info("👈 **Welcome!** Please paste your data in the sidebar to begin the analysis.")
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop",
             caption="Provide your OGTT data to generate your metabolic analysis.",
             use_column_width=True)