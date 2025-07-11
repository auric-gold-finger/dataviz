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
        
        # Find peak insulin value and time
        peak_insulin = df['Insulin'].max()
        peak_time = df.loc[df['Insulin'].idxmax(), 'Time']

        # Pattern I: Normal
        if i120 < i0 and peak_insulin < 100 and peak_time <= 60:
             return "Pattern I: Normal", "Normal insulin response. Insulin peaks early and returns to baseline by 2 hours."

        # Pattern V: Diabetic response with beta-cell failure
        if i0 > 30 and i30 < i0 and i60 < i0 and i120 < i0:
            return "Pattern V: Diabetic Response (Beta-cell Failure)", "Low, flat insulin curve, suggesting severe beta-cell exhaustion. Often seen in established Type 1 or late-stage Type 2 Diabetes."

        # Pattern IV: Irreversible IR / Diabetes
        if i0 >= 30 and i120 > i0:
             return "Pattern IV: Irreversible Insulin Resistance", "High fasting insulin and a sustained high insulin level at 2 hours, indicating severe and established insulin resistance, consistent with Type 2 Diabetes."

        # Pattern II & III are degrees of hyperinsulinemia
        # Pattern III is a delayed peak
        if peak_time > 60 and i120 > i0:
            if i0 < 30:
                 return "Pattern IIIa: Hyperinsulinemia (Delayed Peak)", "Insulin response is delayed, peaking late (at or after 2 hours) and remaining high. This is a clear sign of insulin resistance."
            else:
                 return "Pattern IIIb: Hyperinsulinemia (High Fasting & Delayed Peak)", "High fasting insulin combined with a delayed peak indicates significant, advanced insulin resistance."
        
        # Pattern II: Hyperinsulinemia
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
        
        if peak_glucose < 140 and g120 < 100:
            return "Low Variability", "Excellent glucose control. Glucose levels remain very stable with minimal increase after the challenge."
        
        if peak_time <= 60 and peak_glucose >= 140 and g120 < 140:
             return "Moderate Variability (Classic Spike)", "A moderate, early glucose spike that resolves well below the 2-hour threshold for pre-diabetes. A common pattern."

        if peak_time > 60 and peak_glucose >= 140:
            return "High Variability (Delayed Spike)", "Glucose peaks late in the test, indicating a delayed ability to handle the glucose load. This pattern is associated with a higher risk of impaired glucose tolerance."
        
        if g120 >= 140:
            return "Severe Variability (Impaired Glucose Tolerance)", "Glucose remains high (≥140 mg/dL) at the 2-hour mark, meeting the criteria for Impaired Glucose Tolerance or Diabetes, depending on the value."

        return "Unclassified", "The glucose pattern is unusual and does not fit a standard glucotype."
    except IndexError:
        return "Incomplete Data", "Cannot determine Glucotype due to missing data."


# --- Plotting Function ---
def create_ogtt_plot(df):
    """Creates a dual-axis Plotly chart for Glucose and Insulin."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Glucose trace
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Glucose'],
            name='Glucose (mg/dL)',
            mode='lines+markers',
            marker=dict(color='#1f77b4', size=10),
            line=dict(width=3)
        ),
        secondary_y=False,
    )

    # Add Insulin trace
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Insulin'],
            name='Insulin (µU/mL)',
            mode='lines+markers',
            marker=dict(color='#ff7f0e', size=10),
            line=dict(width=3, dash='dash')
        ),
        secondary_y=True,
    )

    # Add critical glucose lines
    fig.add_hline(y=140, line_dash="dot",
                  annotation_text="IGT Threshold (2h)",
                  annotation_position="bottom right",
                  secondary_y=False, line_color="red")
    fig.add_hline(y=100, line_dash="dot",
                  annotation_text="Normal Fasting Upper Limit",
                  annotation_position="top right",
                  secondary_y=False, line_color="green")


    # Set titles and labels
    fig.update_layout(
        title_text="<b>OGTT Glucose and Insulin Response</b>",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Time (minutes)")
    fig.update_yaxes(title_text="<b>Glucose (mg/dL)</b>", secondary_y=False, color='#1f77b4')
    fig.update_yaxes(title_text="<b>Insulin (µU/mL)</b>", secondary_y=True, color='#ff7f0e')

    return fig

# --- Streamlit UI ---

st.title("🔬 OGTT Analyzer")
st.markdown("""
This application analyzes Oral Glucose Tolerance Test (OGTT) data to provide insights into metabolic health. 
Upload your data to visualize the glucose and insulin curves and calculate key metabolic indices.
**Disclaimer:** This is an educational tool and not a substitute for professional medical advice.
""")

# --- Sidebar for Data Input ---
with st.sidebar:
    st.header("Input OGTT Data")
    st.markdown("Units: Glucose in **mg/dL**, Insulin in **µU/mL**.")
    
    input_method = st.radio(
        "Choose your data input method:",
        ("Manual Input", "Paste Data", "Upload CSV")
    )

    data = None
    df = None
    
    # Standard time points
    time_points = [0, 30, 60, 90, 120]

    if input_method == "Manual Input":
        st.subheader("Manual Entry")
        cols = st.columns(len(time_points))
        glucose_values = []
        insulin_values = []
        for i, t in enumerate(time_points):
            with cols[i]:
                st.markdown(f"**{t} min**")
                glucose_values.append(st.number_input(f"Gluc {t}", key=f"g{t}", min_value=0.0, step=1.0, value=None))
                insulin_values.append(st.number_input(f"Ins {t}", key=f"i{t}", min_value=0.0, step=0.1, value=None))
        
        if all(g is not None and i is not None for g, i in zip(glucose_values, insulin_values)):
            data = {'Time': time_points, 'Glucose': glucose_values, 'Insulin': insulin_values}
            df = pd.DataFrame(data)

    elif input_method == "Paste Data":
        st.subheader("Paste from Spreadsheet")
        st.markdown("Paste data with 3 columns: Time, Glucose, Insulin (no headers).")
        pasted_data = st.text_area("Paste data here", height=150, placeholder="0\t85\t5.0\n30\t155\t50.0\n60\t130\t65.0\n90\t100\t35.0\n120\t90\t15.0")
        if pasted_data:
            try:
                data_io = io.StringIO(pasted_data)
                df = pd.read_csv(data_io, sep=r'\s+', header=None, names=['Time', 'Glucose', 'Insulin'])
                st.success("Data pasted successfully!")
            except Exception as e:
                st.error(f"Error parsing data: {e}. Please ensure it's in the correct format (Time, Glucose, Insulin separated by spaces or tabs).")

    elif input_method == "Upload CSV":
        st.subheader("Upload a CSV file")
        st.markdown("CSV should have 3 columns: 'Time', 'Glucose', 'Insulin'.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if not all(col in df.columns for col in ['Time', 'Glucose', 'Insulin']):
                     st.error("CSV must contain 'Time', 'Glucose', and 'Insulin' columns.")
                     df = None
                else:
                    st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")


# --- Main Panel for Results ---
if df is not None and not df.empty and df.isnull().sum().sum() == 0:
    st.header("Analysis Results")
    
    # --- Data Table and Plot ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Data")
        st.dataframe(df, hide_index=True)
    
    with col2:
        st.plotly_chart(create_ogtt_plot(df), use_container_width=True)

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
    with res_cols[0]:
        st.metric(
            label="Matsuda Index", 
            value=f"{matsuda:.2f}" if not np.isnan(matsuda) else "N/A",
            help="Measures whole-body insulin sensitivity. Higher is better (typically > 4.0 is good)."
        )
    with res_cols[1]:
        st.metric(
            label="HOMA-IR", 
            value=f"{homa_ir:.2f}" if not np.isnan(homa_ir) else "N/A",
            help="Estimates insulin resistance from fasting glucose and insulin. Lower is better (typically < 1.8 is optimal)."
        )
    with res_cols[2]:
        st.metric(
            label="Insulinogenic Index (IGI₃₀)", 
            value=f"{igi:.2f}" if not np.isnan(igi) else "N/A",
            help="Measures early-phase insulin secretion. A lower value may indicate beta-cell dysfunction."
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
    st.info("Please input data in the sidebar to begin analysis.")
    st.image("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop",
             caption="Provide your OGTT data to generate your metabolic analysis.",
             use_column_width=True)
