# -----------------------------------
# Import necessary libraries
# -----------------------------------

import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge
from io import BytesIO

from textwrap import dedent

from Operator import get_random_prediction
from Data_Scientist import benchmark_models


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Transient Stability Limit Predictor",
    layout="wide"
)


# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown(
    dedent(
        """
        <style>

        .pipeline-box {
            border: 1px solid #d9d9d9;
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 8px;
            background-color: rgba(128, 128, 128, 0.08);
        }

        .pipeline-step {
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            padding: 8px;
        }

        .pipeline-arrow {
            text-align: center;
            font-size: 26px;
            line-height: 26px;
            opacity: 0.70;
        }

        .full-pipeline-box {
            border: 1px solid #d9d9d9;
            border-radius: 12px;
            padding: 20px;
            background-color: rgba(128, 128, 128, 0.08);
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# -----------------------------------
# Dataset path
# -----------------------------------

DATA_PATH = "long_df_head.csv"
# DATA_PATH = (
#     r"C:\Users\bernh\Documents\GCU\DSC-580"
#     r"\Milestone 3\long_df_head.csv"
# )


# -----------------------------------
# Initial dataset load
# -----------------------------------

df = pd.read_csv(DATA_PATH)


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

# -------------------------
# Operator session state
# -------------------------

if "operator_df" not in st.session_state:
    st.session_state.operator_df = df.copy()

if "operator_obtain_complete" not in st.session_state:
    st.session_state.operator_obtain_complete = False

if "operator_scrub_complete" not in st.session_state:
    st.session_state.operator_scrub_complete = False

if "operator_explore_complete" not in st.session_state:
    st.session_state.operator_explore_complete = False

if "operator_model_complete" not in st.session_state:
    st.session_state.operator_model_complete = False

if "operator_interpret_complete" not in st.session_state:
    st.session_state.operator_interpret_complete = False

if "operator_data_source" not in st.session_state:
    st.session_state.operator_data_source = "Read File"

if "operator_scrub_method" not in st.session_state:
    st.session_state.operator_scrub_method = "Default"


# -------------------------
# Data Scientist session state
# -------------------------

if "ds_df" not in st.session_state:
    st.session_state.ds_df = df.copy()

if "ds_obtain_complete" not in st.session_state:
    st.session_state.ds_obtain_complete = False

if "ds_scrub_complete" not in st.session_state:
    st.session_state.ds_scrub_complete = False

if "ds_explore_complete" not in st.session_state:
    st.session_state.ds_explore_complete = False

if "ds_model_complete" not in st.session_state:
    st.session_state.ds_model_complete = False

if "ds_interpret_complete" not in st.session_state:
    st.session_state.ds_interpret_complete = False

if "ds_data_source" not in st.session_state:
    st.session_state.ds_data_source = "Read File"

if "ds_scrub_method" not in st.session_state:
    st.session_state.ds_scrub_method = "Default"


# =========================================================
# OPERATOR OSEMN FUNCTIONS
# =========================================================


def operator_obtain_data():
    """
    O - Obtain

    Obtain the data needed by the Operator view.

    In the final application this could represent
    obtaining the current operating condition from
    SCADA, EMS, or another operating-data source.
    """

    if st.session_state.operator_data_source == "Read File":
        operator_df = pd.read_csv(DATA_PATH)
    else:  # Read URL
        url = "https://www.dropbox.com/scl/fi/0msrg6c9i38tg80flf3pg/long_df_head.csv?rlkey=vbebskj89qosadwaxixvdzwiy&st=124rxiep&dl=1"
        operator_df = pd.read_csv(url)

    st.session_state.operator_df = operator_df

    st.session_state.operator_obtain_complete = True

    return operator_df


def operator_scrub_data():
    """
    S - Scrub

    Prepare the data required for the Operator prediction.
    """

    if not st.session_state.operator_obtain_complete:
        operator_obtain_data()

    operator_df = (
        st.session_state.operator_df.copy()
    )

    operator_df = operator_df.drop_duplicates()
    
    # Apply scrubbing method based on selection
    if st.session_state.operator_scrub_method in ["Default", "Remove if Sample Missing"]:
        operator_df = operator_df.dropna(
            subset=[
                "P_remaining_sum",
                "P_diff_target0_est_pred"
            ]
        )
    # "No Scrubbing" does nothing

    st.session_state.operator_clean_df = (
        operator_df
    )

    st.session_state.operator_scrub_complete = True

    return operator_df


def operator_explore_data():
    """
    E - Explore

    Examine the operating condition before prediction.
    """

    if not st.session_state.operator_scrub_complete:
        operator_scrub_data()

    operator_df = (
        st.session_state.operator_clean_df
    )


    st.session_state.operator_explore_df = (
        operator_df
    )

    st.session_state.operator_explore_complete = True

    return operator_df


def operator_model_data():
    """
    M - Model

    Apply the trained prediction model to an operating
    condition.
    """

    if not st.session_state.operator_explore_complete:
        operator_explore_data()

    operator_df = (
        st.session_state.operator_explore_df
    )

    prediction, current_flow = (
        get_random_prediction(
            operator_df
        )
    )

    st.session_state.operator_prediction = (
        prediction
    )

    st.session_state.operator_current_flow = (
        current_flow
    )

    st.session_state.operator_model_complete = True

    return prediction, current_flow


def operator_interpret_results():
    """
    N - iNterpret

    Translate the model prediction into information useful
    to the system Operator.
    """

    if not st.session_state.operator_model_complete:
        operator_model_data()

    prediction = (
        st.session_state.operator_prediction
    )

    current_flow = (
        st.session_state.operator_current_flow
    )

    operating_margin = (
        prediction - current_flow
    )

    percent_of_limit = (
        current_flow / prediction
    ) * 100

    # Determine operating condition
    if percent_of_limit > 95:

        status = "Critical"

    elif percent_of_limit > 85:

        status = "Caution"

    else:

        status = "Normal"

    st.session_state.operator_operating_margin = (
        operating_margin
    )

    st.session_state.operator_percent_of_limit = (
        percent_of_limit
    )

    st.session_state.operator_status = (
        status
    )

    st.session_state.operator_interpret_complete = True

    return (
        operating_margin,
        percent_of_limit,
        status
    )


def run_operator_pipeline():
    """
    Run the complete Operator OSEMN pipeline.
    """

    operator_obtain_data()

    operator_scrub_data()

    operator_explore_data()

    operator_model_data()

    operator_interpret_results()


# =========================================================
# DATA SCIENTIST OSEMN FUNCTIONS
# =========================================================


def ds_obtain_data():
    """
    O - Obtain

    Obtain the dataset used for model development,
    validation, and benchmarking.
    """

    if st.session_state.ds_data_source == "Read File":
        ds_df = pd.read_csv(DATA_PATH)
    else:  # Read URL
        url = "https://www.dropbox.com/scl/fi/0msrg6c9i38tg80flf3pg/long_df_head.csv?rlkey=vbebskj89qosadwaxixvdzwiy&st=124rxiep&dl=1"
        ds_df = pd.read_csv(url)

    st.session_state.ds_df = ds_df

    st.session_state.ds_obtain_complete = True

    return ds_df


def ds_scrub_data():
    """
    S - Scrub

    Clean and preprocess the dataset before analysis
    and machine-learning modeling.
    """

    if not st.session_state.ds_obtain_complete:
        ds_obtain_data()

    ds_df = (
        st.session_state.ds_df.copy()
    )

    ds_df = ds_df.drop_duplicates()
    
    # Apply scrubbing method based on selection
    scrub_method = st.session_state.ds_scrub_method
    
    if scrub_method in ["Default", "Remove Missing Samples"]:
        # Remove missing values
        ds_df = ds_df.dropna(
            subset=[
                "P_remaining_sum",
                "P_diff_target0_est_pred"
            ]
        )
    
    if scrub_method in ["Default", "Remove Outliers"]:
        # Remove outliers using IQR method for numeric columns
        # numeric_cols = ds_df.select_dtypes(include=["number"]).columns
        # for col in numeric_cols:
        #     Q1 = ds_df[col].quantile(0.25)
        #     Q3 = ds_df[col].quantile(0.75)
        #     IQR = Q3 - Q1
        #     lower_bound = Q1 - 1.5 * IQR
        #     upper_bound = Q3 + 1.5 * IQR
        #     ds_df = ds_df[(ds_df[col] >= lower_bound) & (ds_df[col] <= upper_bound)]
    
    # "No Scrubbing" does nothing

        st.session_state.ds_clean_df = (
            ds_df
        )

    st.session_state.ds_scrub_complete = True

    return ds_df


def ds_explore_data():
    """
    E - Explore

    Perform exploratory data analysis before modeling.
    """

    if not st.session_state.ds_scrub_complete:
        ds_scrub_data()

    ds_df = (
        st.session_state.ds_clean_df
    )

    numeric_df = (
        ds_df.select_dtypes(
            include="number"
        )
    )

    if not numeric_df.empty:

        st.session_state.ds_summary_stats = (
            numeric_df.describe()
        )

        st.session_state.ds_correlations = (
            numeric_df.corr()
        )

    st.session_state.ds_explore_complete = True

    return ds_df


def ds_model_data():
    """
    M - Model

    Benchmark the trained machine-learning models.
    """

    if not st.session_state.ds_explore_complete:
        ds_explore_data()

    ds_df = (
        st.session_state.ds_clean_df
    )

    results = benchmark_models(
        ds_df,
        target_column="P_diff_target0_est_pred",
        sample_size=1000
    )

    st.session_state.benchmark_results = (
        results
    )

    st.session_state.ds_model_complete = True

    return results


def ds_interpret_results():
    """
    N - iNterpret

    Interpret benchmark metrics and identify the
    strongest-performing models.
    """

    if not st.session_state.ds_model_complete:
        ds_model_data()

    results = (
        st.session_state.benchmark_results
    )

    if not results.empty:

        best_r2_index = (
            results["R²"].idxmax()
        )

        fastest_index = (
            results[
                "Single Prediction Time (ms)"
            ].idxmin()
        )

        st.session_state.ds_best_accuracy_model = (
            results.loc[
                best_r2_index,
                "Model"
            ]
        )

        st.session_state.ds_fastest_model = (
            results.loc[
                fastest_index,
                "Model"
            ]
        )

    st.session_state.ds_interpret_complete = True

    return results


def run_ds_pipeline():
    """
    Run the complete Data Scientist OSEMN pipeline.
    """

    ds_obtain_data()

    ds_scrub_data()

    ds_explore_data()

    ds_model_data()

    ds_interpret_results()

# =========================================================
# PIPELINE RESET FUNCTIONS
# =========================================================


def reset_operator_pipeline():
    """
    Reset the Operator pipeline status and remove
    results generated by the previous pipeline run.
    """

    # Reset completion indicators
    st.session_state.operator_obtain_complete = False
    st.session_state.operator_scrub_complete = False
    st.session_state.operator_explore_complete = False
    st.session_state.operator_model_complete = False
    st.session_state.operator_interpret_complete = False

    # Remove intermediate/results data
    operator_keys = [
        "operator_clean_df",
        "operator_explore_df",
        "operator_prediction",
        "operator_current_flow",
        "operator_operating_margin",
        "operator_percent_of_limit",
        "operator_status"
    ]

    for key in operator_keys:
        st.session_state.pop(key, None)



def reset_ds_pipeline():
    """
    Reset the Data Scientist pipeline status and remove
    results generated by the previous pipeline run.
    """

    # Reset completion indicators
    st.session_state.ds_obtain_complete = False
    st.session_state.ds_scrub_complete = False
    st.session_state.ds_explore_complete = False
    st.session_state.ds_model_complete = False
    st.session_state.ds_interpret_complete = False

    # Remove intermediate/results data
    ds_keys = [
        "ds_clean_df",
        "ds_summary_stats",
        "ds_correlations",
        "benchmark_results",
        "ds_best_accuracy_model",
        "ds_fastest_model"
    ]

    for key in ds_keys:
        st.session_state.pop(key, None)

# =========================================================
# SHARED DISPLAY FUNCTIONS
# =========================================================


def pipeline_arrow():
    """
    Display pipeline-flow arrow.
    """

    st.markdown(
        dedent(
            """
            <div class="pipeline-arrow">
                ↓
            </div>
            """
        ),
        unsafe_allow_html=True
    )


def get_status(complete):
    """
    Return graphical pipeline completion indicator.
    """

    if complete:
        return "✅"

    return "⬜"


# =========================================================
# OPERATOR PIPELINE CONTROLS
# =========================================================


def display_operator_pipeline_controls():

    st.subheader(
        "Pipeline Controls"
    )

    full_col, steps_col = st.columns(
        [1, 2],
        gap="large"
    )

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with full_col:

        st.markdown(
            dedent(
                """
                <div class="full-pipeline-box">

                <h3 style="text-align:center;">
                    Operator Pipeline
                </h3>

                <p style="text-align:center;">
                    Run the complete Operator OSEMN
                    workflow from operating data through
                    transient stability interpretation.
                </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "▶ Run Entire Operator Pipeline",
            key="operator_full_pipeline",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Running Operator pipeline..."
            ):

                run_operator_pipeline()

            st.success(
                "Operator pipeline completed."
            )

        if st.button(
            "↻ Reset Operator Pipeline",
            key="operator_reset_pipeline",
            use_container_width=True
        ):

            reset_operator_pipeline()

            st.rerun()

        st.markdown(
            "### Pipeline Status"
        )

        st.write(
            f"{get_status(st.session_state.operator_obtain_complete)} "
            "Obtain"
        )

        st.write(
            f"{get_status(st.session_state.operator_scrub_complete)} "
            "Scrub"
        )

        st.write(
            f"{get_status(st.session_state.operator_explore_complete)} "
            "Explore"
        )

        st.write(
            f"{get_status(st.session_state.operator_model_complete)} "
            "Model"
        )

        st.write(
            f"{get_status(st.session_state.operator_interpret_complete)} "
            "iNterpret"
        )


    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with steps_col:

        st.markdown(
            dedent(
                """
                <div class="pipeline-box">

                <h3 style="text-align:center;">
                    Operator OSEMN Workflow
                </h3>

                <p style="text-align:center;">
                    Execute each stage individually.
                </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # OBTAIN

        obtain_status = get_status(
            st.session_state.operator_obtain_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {obtain_status} 1. Obtain Operating Data
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.session_state.operator_data_source = st.selectbox(
            "Data Source:",
            ["Read File", "Read URL"],
            index=0 if st.session_state.operator_data_source == "Read File" else 1,
            key="operator_data_source_select"
        )

        if st.button(
            "📥 Obtain Operating Condition",
            key="operator_obtain",
            use_container_width=True
        ):

            operator_obtain_data()

            st.success(
                "Operating data obtained."
            )

        pipeline_arrow()


        # SCRUB

        scrub_status = get_status(
            st.session_state.operator_scrub_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {scrub_status} 2. Scrub Operating Data
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.session_state.operator_scrub_method = st.selectbox(
            "Scrubbing Method:",
            ["Default", "Remove if Sample Missing", "No Scrubbing"],
            index=["Default", "Remove if Sample Missing", "No Scrubbing"].index(st.session_state.operator_scrub_method),
            key="operator_scrub_method_select"
        )

        if st.button(
            "🧹 Validate and Prepare Data",
            key="operator_scrub",
            use_container_width=True
        ):

            operator_scrub_data()

            st.success(
                "Operating data prepared."
            )

        pipeline_arrow()


        # EXPLORE

        explore_status = get_status(
            st.session_state.operator_explore_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {explore_status} 3. Explore Operating Condition
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "🔎 Review Operating Condition",
            key="operator_explore",
            use_container_width=True
        ):

            operator_explore_data()

            st.success(
                "Operating condition reviewed."
            )

        pipeline_arrow()


        # MODEL

        model_status = get_status(
            st.session_state.operator_model_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {model_status} 4. Model Stability Limit
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "🤖 Predict Stability Limit",
            key="operator_model",
            use_container_width=True
        ):

            operator_model_data()

            st.success(
                "Stability limit predicted."
            )

        pipeline_arrow()


        # INTERPRET

        interpret_status = get_status(
            st.session_state.operator_interpret_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {interpret_status} 5. iNterpret Operating Margin
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "📊 Calculate Operating Margin",
            key="operator_interpret",
            use_container_width=True
        ):

            operator_interpret_results()

            st.success(
                "Operating margin interpreted."
            )

        pipeline_arrow()


        # PDF EXPORT

        if st.session_state.operator_model_complete:

            prediction = st.session_state.operator_prediction
            current_flow = st.session_state.operator_current_flow
            operating_margin = st.session_state.operator_operating_margin
            percent_of_limit = st.session_state.operator_percent_of_limit
            status = st.session_state.operator_status

            if percent_of_limit > 95:
                gauge_color = "#d9534f"
            elif percent_of_limit > 85:
                gauge_color = "#f0ad4e"
            else:
                gauge_color = "#5cb85c"

            with st.expander(
                "Export Prediction Results as PDF"
            ):

                pdf_buffer = BytesIO()

                with PdfPages(pdf_buffer) as pdf:

                    # Figure 1 - Summary + gauge chart
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.patch.set_facecolor("white")
                    fig.suptitle(
                        "Operator View Report",
                        fontsize=22,
                        fontweight="bold",
                        y=0.97,
                    )

                    summary_rows = [
                        ("Data Source", st.session_state.get("operator_data_source", "Read File")),
                        ("Scrub Method", st.session_state.get("operator_scrub_method", "Default")),
                        ("Current Flow", f"{current_flow:,.0f} MW"),
                        ("Predicted Stability Limit", f"{prediction:,.0f} MW"),
                        ("Remaining Margin", f"{operating_margin:,.0f} MW"),
                        ("Percent of Limit", f"{percent_of_limit:.1f}%"),
                        ("Operating Status", status),
                    ]

                    summary_x = 0.08
                    summary_y = 0.78
                    for label, value in summary_rows:
                        fig.text(
                            summary_x,
                            summary_y,
                            f"{label}: {value}",
                            fontsize=11,
                            va="center",
                        )
                        summary_y -= 0.05

                    gauge_ax = fig.add_axes([0.12, 0.22, 0.76, 0.42])
                    gauge_ax.set_aspect("equal")
                    gauge_ax.axis("off")

                    gauge_bg = Wedge(
                        center=(0, 0),
                        r=1.15,
                        theta1=0,
                        theta2=180,
                        width=0.32,
                        facecolor="#e6e6e6",
                        edgecolor="none",
                    )
                    gauge_ax.add_patch(gauge_bg)

                    level = min(max(percent_of_limit, 0), 100)
                    sweep = 180 * level / 100
                    gauge_fill = Wedge(
                        center=(0, 0),
                        r=1.15,
                        theta1=180,
                        theta2=180 - sweep,
                        width=0.32,
                        facecolor=gauge_color,
                        edgecolor="none",
                    )
                    gauge_ax.add_patch(gauge_fill)

                    for tick in [0, 25, 50, 75, 100]:
                        angle = 180 - tick * 1.8
                        rad = math.radians(angle)
                        inner = 0.82
                        outer = 1.08
                        x1, y1 = inner * math.cos(rad), inner * math.sin(rad)
                        x2, y2 = outer * math.cos(rad), outer * math.sin(rad)
                        gauge_ax.plot([x1, x2], [y1, y2], color="black", lw=1)
                        label_r = 1.25
                        lx, ly = label_r * math.cos(rad), label_r * math.sin(rad)
                        gauge_ax.text(lx, ly, f"{tick}%", ha="center", va="center", fontsize=9)

                    needle_angle = 180 - (level * 1.8)
                    needle_rad = math.radians(needle_angle)
                    nx, ny = 0.9 * math.cos(needle_rad), 0.9 * math.sin(needle_rad)
                    gauge_ax.plot([0, nx], [0, ny], color="black", lw=2.5)
                    gauge_ax.plot(0, 0, "o", color="black", markersize=6)

                    gauge_ax.text(0, 1.45, f"{level:.1f}% of limit", ha="center", va="center", fontsize=12, fontweight="bold")
                    gauge_ax.set_xlim(-1.4, 1.4)
                    gauge_ax.set_ylim(-0.25, 1.7)

                    pdf.savefig(fig)
                    plt.close(fig)

                    # Figure 2 - Metrics table
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.axis("off")

                    table_data = [
                        ["Metric", "Value"],
                        ["Current Flow", f"{current_flow:,.0f} MW"],
                        ["Predicted Stability Limit", f"{prediction:,.0f} MW"],
                        ["Remaining Margin", f"{operating_margin:,.0f} MW"],
                        ["Percent of Limit", f"{percent_of_limit:.1f}%"],
                        ["Operating Status", status],
                    ]

                    table = ax.table(
                        cellText=table_data,
                        colLabels=None,
                        cellLoc="left",
                        loc="center",
                        bbox=[0, 0, 1, 1],
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(11)
                    table.scale(1.2, 1.6)
                    ax.set_title("Prediction Metrics", fontsize=14, pad=18)

                    pdf.savefig(fig)
                    plt.close(fig)

                pdf_buffer.seek(0)

                st.download_button(
                    "📥 Download PDF Report",
                    pdf_buffer,
                    "prediction_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )

        else:

            st.info(
                "Run the Operator Model stage or the "
                "entire Operator pipeline to generate "
                "a transient stability prediction."
            )


# =========================================================
# DATA SCIENTIST PIPELINE CONTROLS
# =========================================================


def display_ds_pipeline_controls():

    st.subheader(
        "Pipeline Controls"
    )

    full_col, steps_col = st.columns(
        [1, 2],
        gap="large"
    )

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with full_col:

        st.markdown(
            dedent(
                """
                <div class="full-pipeline-box">

                <h3 style="text-align:center;">
                    Data Scientist Pipeline
                </h3>

                <p style="text-align:center;">
                    Run the complete model-development
                    and evaluation workflow.
                </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "▶ Run Entire Data Scientist Pipeline",
            key="ds_full_pipeline",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Running Data Scientist pipeline..."
            ):

                run_ds_pipeline()

            st.success(
                "Data Scientist pipeline completed."
            )

        if st.button(
            "↻ Reset Data Scientist Pipeline",
            key="ds_reset_pipeline",
            use_container_width=True
        ):

            reset_ds_pipeline()

            st.rerun()

        st.markdown(
            "### Pipeline Status"
        )

        st.write(
            f"{get_status(st.session_state.ds_obtain_complete)} "
            "Obtain"
        )

        st.write(
            f"{get_status(st.session_state.ds_scrub_complete)} "
            "Scrub"
        )

        st.write(
            f"{get_status(st.session_state.ds_explore_complete)} "
            "Explore"
        )

        st.write(
            f"{get_status(st.session_state.ds_model_complete)} "
            "Model"
        )

        st.write(
            f"{get_status(st.session_state.ds_interpret_complete)} "
            "iNterpret"
        )


    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with steps_col:

        st.markdown(
            dedent(
                """
                <div class="pipeline-box">

                <h3 style="text-align:center;">
                    Data Scientist OSEMN Workflow
                </h3>

                <p style="text-align:center;">
                    Execute each development stage
                    individually.
                </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # OBTAIN

        obtain_status = get_status(
            st.session_state.ds_obtain_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {obtain_status} 1. Obtain Training Data
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.session_state.ds_data_source = st.selectbox(
            "Data Source:",
            ["Read File", "Read URL"],
            index=0 if st.session_state.ds_data_source == "Read File" else 1,
            key="ds_data_source_select"
        )

        if st.button(
            "📥 Obtain Training Data",
            key="ds_obtain",
            use_container_width=True
        ):

            ds_obtain_data()

            st.success(
                "Training data obtained."
            )

        pipeline_arrow()


        # SCRUB

        scrub_status = get_status(
            st.session_state.ds_scrub_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {scrub_status} 2. Scrub and Prepare Data
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.session_state.ds_scrub_method = st.selectbox(
            "Scrubbing Method:",
            ["Default", "Remove Missing Samples", "Remove Outliers", "No Scrubbing"],
            index=["Default", "Remove Missing Samples", "Remove Outliers", "No Scrubbing"].index(st.session_state.ds_scrub_method),
            key="ds_scrub_method_select"
        )

        if st.button(
            "🧹 Clean and Preprocess Data",
            key="ds_scrub",
            use_container_width=True
        ):

            ds_scrub_data()

            st.success(
                "Training data prepared."
            )

        pipeline_arrow()


        # EXPLORE

        explore_status = get_status(
            st.session_state.ds_explore_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {explore_status} 3. Explore Data
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "🔎 Run Exploratory Analysis",
            key="ds_explore",
            use_container_width=True
        ):

            ds_explore_data()

            st.success(
                "Exploratory analysis completed."
            )

        pipeline_arrow()


        # MODEL

        model_status = get_status(
            st.session_state.ds_model_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {model_status} 4. Model and Benchmark
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "🤖 Benchmark Models",
            key="ds_model",
            use_container_width=True
        ):

            with st.spinner(
                "Benchmarking models..."
            ):

                ds_model_data()

            st.success(
                "Model benchmarking completed."
            )

        pipeline_arrow()


        # INTERPRET

        interpret_status = get_status(
            st.session_state.ds_interpret_complete
        )

        st.markdown(
            dedent(
                f"""
                <div class="pipeline-step">
                    {interpret_status} 5. iNterpret Model Results
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        if st.button(
            "📊 Interpret Model Performance",
            key="ds_interpret",
            use_container_width=True
        ):

            ds_interpret_results()

            st.success(
                "Model results interpreted."
            )


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title(
    "Transient Stability Limit Predictor"
)

st.markdown(
    """
    Grand Canyon University &nbsp;&nbsp;&nbsp;&nbsp;
    DSC-580 &nbsp;&nbsp;&nbsp;&nbsp;
    Milestone 3 &nbsp;&nbsp;&nbsp;&nbsp;
    Douglas Bernhoft &nbsp;&nbsp;&nbsp;&nbsp;
    Revision 8/9/2026
    """,
    unsafe_allow_html=True
)


# =========================================================
# CREATE TABS
# =========================================================

operator_tab, data_scientist_tab = st.tabs(
    [
        "Operator View",
        "Data Scientist View"
    ]
)


# =========================================================
# OPERATOR VIEW
# =========================================================

with operator_tab:

    st.header(
        "Operator View"
    )

    st.markdown(
        """
        The Operator workflow uses current operating data
        to estimate the transient stability limit and
        determine the remaining operating margin.
        """
    )

    # -----------------------------------
    # Operator OSEMN controls
    # -----------------------------------

    display_operator_pipeline_controls()

    st.divider()


    # -----------------------------------
    # Operator Results
    # -----------------------------------

    st.subheader(
        "Transient Stability Prediction"
    )

    if st.session_state.operator_model_complete:

        prediction = (
            st.session_state.operator_prediction
        )

        current_flow = (
            st.session_state.operator_current_flow
        )

        # Calculate interpretation if it has not
        # already been performed.

        if not st.session_state.operator_interpret_complete:

            operator_interpret_results()

        operating_margin = (
            st.session_state.operator_operating_margin
        )

        percent_of_limit = (
            st.session_state.operator_percent_of_limit
        )

        status = (
            st.session_state.operator_status
        )


        # -----------------------------------
        # Determine gauge color
        # -----------------------------------

        if percent_of_limit > 95:

            gauge_color = "red"

        elif percent_of_limit > 85:

            gauge_color = "yellow"

        else:

            gauge_color = "green"


        # -----------------------------------
        # Output columns
        # -----------------------------------

        col1, col2 = st.columns(
            [2, 1]
        )


        # -----------------------------------
        # Column 1 - Gauge
        # -----------------------------------

        with col1:

            fig = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=current_flow,

                    number={
                        "suffix": " MW"
                    },

                    title={
                        "text":
                        "Interface Loading"
                    },

                    gauge={

                        "axis": {
                            "range": [
                                0,
                                prediction * 1.10
                            ]
                        },

                        "bar": {
                            "color":
                            gauge_color
                        },

                        "threshold": {

                            "line": {
                                "color": "red",
                                "width": 5
                            },

                            "thickness": 0.8,

                            "value":
                            prediction
                        }
                    }
                )
            )

            fig.update_layout(
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # -----------------------------------
        # Column 2 - Metrics
        # -----------------------------------

        with col2:

            st.metric(
                "Current Flow",
                f"{current_flow:,.0f} MW"
            )

            st.metric(
                "Predicted Stability Limit",
                f"{prediction:,.0f} MW"
            )

            st.metric(
                "Remaining Margin",
                f"{operating_margin:,.0f} MW"
            )

            st.metric(
                "Percent of Limit",
                f"{percent_of_limit:.1f}%"
            )

            st.metric(
                "Operating Status",
                status
            )

    else:

        st.info(
            "Run the Operator Model stage or the "
            "entire Operator pipeline to generate "
            "a transient stability prediction."
        )


# =========================================================
# DATA SCIENTIST VIEW
# =========================================================

with data_scientist_tab:

    st.header(
        "Data Scientist View"
    )

    st.markdown(
        """
        The Data Scientist workflow obtains, prepares,
        explores, models, and evaluates the transient
        stability dataset.
        """
    )


    # -----------------------------------
    # Data Scientist OSEMN controls
    # -----------------------------------

    display_ds_pipeline_controls()

    st.divider()


    # -----------------------------------
    # Exploratory Results
    # -----------------------------------

    if st.session_state.ds_explore_complete:

        with st.expander(
            "Exploratory Data Analysis"
        ):

            if "ds_summary_stats" in st.session_state:

                st.subheader(
                    "Summary Statistics"
                )

                st.dataframe(
                    st.session_state.ds_summary_stats,
                    use_container_width=True
                )


    # -----------------------------------
    # Benchmark Results
    # -----------------------------------

    st.subheader(
        "Model Benchmark Results"
    )

    if (
        "benchmark_results"
        in st.session_state
        and
        st.session_state.ds_model_complete
    ):

        results = (
            st.session_state.benchmark_results
        )

        col1, col2 = st.columns(
            [2, 1]
        )


        # ===================================
        # COLUMN 1 - SCATTER PLOT
        # ===================================

        with col1:

            fig = px.scatter(

                results,

                x="Single Prediction Time (ms)",

                y="R²",

                text="Model",

                hover_name="Model",

                hover_data={

                    "R²": ":.4f",

                    "RMSE": ":.2f",

                    "MAE": ":.2f",

                    "Single Prediction Time (ms)":
                        ":.4f"
                }
            )


            fig.update_traces(

                marker={
                    "size": 14
                },

                textposition="top center"
            )


            fig.update_layout(

                title=(
                    "Model Accuracy vs. "
                    "Prediction Speed"
                ),

                xaxis_title=(
                    "Average Prediction "
                    "Time (ms)"
                ),

                yaxis_title="R²"
            )


            event = st.plotly_chart(

                fig,

                use_container_width=True,

                on_select="rerun",

                selection_mode=[
                    "points",
                    "box",
                    "lasso"
                ],

                key="model_scatter"
            )


        # ===================================
        # GET SELECTED POINTS
        # ===================================

        selected_indices = []

        if event.selection.points:

            selected_indices = [

                point["point_index"]

                for point
                in event.selection.points
            ]


        # ===================================
        # COLUMN 2 - MODEL TABLE
        # ===================================

        with col2:

            st.subheader(
                "Selected Model Metrics"
            )

            if selected_indices:

                selected_models = (
                    results.iloc[
                        selected_indices
                    ]
                )

                st.dataframe(

                    selected_models[
                        [
                            "Model",
                            "R²",
                            "RMSE",
                            "MAE",
                            "Single Prediction Time (ms)"
                        ]
                    ],

                    use_container_width=True,

                    hide_index=True
                )

            else:

                st.info(
                    "Click, box-select, or lasso "
                    "models in the chart to view "
                    "their metrics."
                )


        # -----------------------------------
        # Interpretation results
        # -----------------------------------

        if st.session_state.ds_interpret_complete:

            st.divider()

            st.subheader(
                "Model Interpretation"
            )

            interpretation_col1, interpretation_col2 = (
                st.columns(2)
            )

            with interpretation_col1:

                st.metric(
                    "Highest R² Model",
                    st.session_state.get(
                        "ds_best_accuracy_model",
                        "N/A"
                    )
                )

            with interpretation_col2:

                st.metric(
                    "Fastest Prediction Model",
                    st.session_state.get(
                        "ds_fastest_model",
                        "N/A"
                    )
                )

    else:

        st.info(
            "Run the Data Scientist Model stage or "
            "the entire Data Scientist pipeline to "
            "generate benchmark results."
        )

def build_operator_pdf_report():
    """
    Build a clear PDF report for the current Operator view.
    The report includes the selected options and the result gauge.
    """

    if not st.session_state.get("operator_model_complete", False):
        return b""

    data_source = st.session_state.get("operator_data_source", "Read File")
    scrub_method = st.session_state.get("operator_scrub_method", "Default")
    prediction = float(st.session_state.get("operator_prediction", 0.0))
    current_flow = float(st.session_state.get("operator_current_flow", 0.0))
    operating_margin = float(st.session_state.get("operator_operating_margin", 0.0))
    percent_of_limit = float(st.session_state.get("operator_percent_of_limit", 0.0))
    status = st.session_state.get("operator_status", "Unknown")

    if percent_of_limit > 95:
        gauge_color = "#d9534f"
    elif percent_of_limit > 85:
        gauge_color = "#f0ad4e"
    else:
        gauge_color = "#5cb85c"

    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle("Operator View Report", fontsize=22, fontweight="bold", y=0.97)

        summary_rows = [
            ("Data Source", data_source),
            ("Scrub Method", scrub_method),
            ("Current Flow", f"{current_flow:,.0f} MW"),
            ("Predicted Stability Limit", f"{prediction:,.0f} MW"),
            ("Remaining Margin", f"{operating_margin:,.0f} MW"),
            ("Percent of Limit", f"{percent_of_limit:.1f}%"),
            ("Operating Status", status),
        ]

        summary_x = 0.08
        summary_y = 0.82
        for label, value in summary_rows:
            fig.text(summary_x, summary_y, f"{label}: {value}", fontsize=11, va="center")
            summary_y -= 0.05

        gauge_ax = fig.add_axes([0.12, 0.22, 0.76, 0.42])
        gauge_ax.set_aspect("equal")
        gauge_ax.axis("off")

        bg = Wedge((0, 0), 1.15, 0, 180, width=0.32, facecolor="#e6e6e6", edgecolor="none")
        gauge_ax.add_patch(bg)

        level = min(max(percent_of_limit, 0), 100)
        sweep = 180 * level / 100
        fill = Wedge((0, 0), 1.15, 180, 180 - sweep, width=0.32, facecolor=gauge_color, edgecolor="none")
        gauge_ax.add_patch(fill)

        for tick in [0, 25, 50, 75, 100]:
            angle_deg = 180 - tick * 1.8
            angle_rad = math.radians(angle_deg)
            x1, y1 = 0.82 * math.cos(angle_rad), 0.82 * math.sin(angle_rad)
            x2, y2 = 1.08 * math.cos(angle_rad), 1.08 * math.sin(angle_rad)
            gauge_ax.plot([x1, x2], [y1, y2], color="black", lw=1)
            lx, ly = 1.25 * math.cos(angle_rad), 1.25 * math.sin(angle_rad)
            gauge_ax.text(lx, ly, f"{tick}%", ha="center", va="center", fontsize=9)

        needle_angle = math.radians(180 - (level * 1.8))
        nx = 0.9 * math.cos(needle_angle)
        ny = 0.9 * math.sin(needle_angle)
        gauge_ax.plot([0, nx], [0, ny], color="black", lw=2.5)
        gauge_ax.plot(0, 0, "o", color="black", markersize=6)
        gauge_ax.text(0, 1.45, f"{level:.1f}% of limit", ha="center", va="center", fontsize=12, fontweight="bold")

        gauge_ax.set_xlim(-1.4, 1.4)
        gauge_ax.set_ylim(-0.25, 1.7)

        pdf.savefig(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.axis("off")
        table_data = [
            ["Metric", "Value"],
            ["Current Flow", f"{current_flow:,.0f} MW"],
            ["Predicted Stability Limit", f"{prediction:,.0f} MW"],
            ["Remaining Margin", f"{operating_margin:,.0f} MW"],
            ["Percent of Limit", f"{percent_of_limit:.1f}%"],
            ["Operating Status", status],
        ]
        table = ax2.table(cellText=table_data, colLabels=None, cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.6)
        ax2.set_title("Prediction Metrics", fontsize=14, pad=18)
        pdf.savefig(fig2)
        plt.close(fig2)

    buffer.seek(0)
    return buffer.getvalue()
