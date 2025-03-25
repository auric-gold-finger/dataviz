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
            textfont=dict(
                family="Avenir",
                size=annotation_size,
                color="black"
            ),
            name=f"{current_date}",
            showlegend=(row == 1),  # Only show in legend for first plot
            legendgroup=current_date
        ),
        row=row, col=1
    )
    
    # Add shading between current line and reference where current > reference
    if show_shading:
        # For proper shading, we need to create fill areas only where current is above reference
        # We'll create a combined array with None values separating segments
        x_combined = []
        y_current = []
        y_reference = []
        
        for i in range(len(time)):
            if current_values[i] > reference_values[i]:
                # This point is above reference, include it
                x_combined.append(time[i])
                y_current.append(current_values[i])
                y_reference.append(reference_values[i])
                
                # If this is the last point or the next point is below reference,
                # we need to close the segment with None values
                if i == len(time) - 1 or current_values[i+1] <= reference_values[i+1]:
                    x_combined.append(None)
                    y_current.append(None)
                    y_reference.append(None)
            else:
                # If previous point was above reference, start a new segment
                if i > 0 and current_values[i-1] > reference_values[i-1]:
                    # We just crossed below, add the crossing point for a clean fill
                    # Find where the lines cross between this point and previous point
                    if current_values[i] != current_values[i-1]:  # Avoid division by zero
                        # Parametric value where lines cross
                        t = (reference_values[i-1] - current_values[i-1]) / (current_values[i] - current_values[i-1] - (reference_values[i] - reference_values[i-1]))
                        if 0 <= t <= 1:  # Valid crossing point
                            cross_x = time[i-1] + t * (time[i] - time[i-1])
                            cross_y = reference_values[i-1] + t * (reference_values[i] - reference_values[i-1])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
                            x_combined.append(None)
                            y_current.append(None)
                            y_reference.append(None)
                
                # If the next point will be above reference, start including points now
                if i < len(time) - 1 and current_values[i+1] > reference_values[i+1]:
                    # We're about to cross above, add the crossing point for a clean fill
                    if current_values[i+1] != current_values[i]:  # Avoid division by zero
                        # Parametric value where lines cross
                        t = (reference_values[i] - current_values[i]) / (current_values[i+1] - current_values[i] - (reference_values[i+1] - reference_values[i]))
                        if 0 <= t <= 1:  # Valid crossing point
                            cross_x = time[i] + t * (time[i+1] - time[i])
                            cross_y = reference_values[i] + t * (reference_values[i+1] - reference_values[i])
                            x_combined.append(cross_x)
                            y_current.append(cross_y)
                            y_reference.append(cross_y)
        
        # Only add shading if we have points to shade
        if len(x_combined) > 0:
            # Add reference line as lower bound
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
            
            # Add current line as upper bound with fill
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
                    showlegend=(row == 1),  # Only show in legend for first plot
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
                showlegend=(row == 1),  # Only show reference in legend once
                legendgroup='reference'
            ),
            row=row, col=1
        )

# Add traces
add_traces(glucose_df, 1, "Glucose")
add_traces(insulin_df, 2, "Insulin")

    # Add proper subplot titles after figure creation
fig.update_layout(
    height=1200,  # Increased height
    showlegend=True,
    template='plotly_white',
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
    paper_bgcolor='rgba(0,0,0,0)',
    # Add proper titles for each subplot with adjusted positions
    annotations=[
        dict(
            text=f'Glucose Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.12,  # Moved up more to increase space below title
            showarrow=False
        ),
        dict(
            text=f'Insulin Response to 75g Dextrose ({dates[0]} vs {dates[1] if len(dates) > 1 else "Reference"})',
            font=dict(family="Cormorant Garamond", size=28, color="Black"),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.48,  # Moved up to avoid overlap
            showarrow=False
        )
    ],
    legend=dict(
        yanchor="top",
        y=1.15,  # Moved lower from the top to create more space
        xanchor="center",
        x=0.5,
        font=dict(family="Avenir"),
        orientation="h"
    ),
    font=dict(family="Avenir"),
    margin=dict(t=200, r=50, b=50, l=50)  # Further increased top margin
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

# Update y-axes with calculated ranges
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

# Display charts in a container
st.markdown('<p class="subtitle">Visualization</p>', unsafe_allow_html=True)
with st.container():
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button for PNG if kaleido is installed
    if KALEIDO_INSTALLED:
        if st.button("Download Plots as PNG"):
            # Convert plot to PNG
            img_bytes = fig.to_image(format="png", width=1200, height=1200, scale=2)
            
            # Create download button
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