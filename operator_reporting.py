# -----------------------------------
# Operator View Reporting Module
# -----------------------------------

import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge
from io import BytesIO
import matplotlib.pyplot as plt
from textwrap import dedent


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

    from Operator import get_random_prediction
    
    DATA_PATH = "long_df_head.csv"
    # DATA_PATH = (
    #     r"C:\Users\bernh\Documents\GCU\DSC-580"
    #     r"\Milestone 3\long_df_head.csv"
    # )

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

    from Operator import get_random_prediction

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
# OPERATOR PIPELINE RESET
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


# =========================================================
# OPERATOR DISPLAY FUNCTIONS
# =========================================================


def get_status(complete):
    """
    Return graphical pipeline completion indicator.
    """

    if complete:
        return "✅"

    return "⬜"


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

                pdf_buffer = build_operator_pdf_report()

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


def display_operator_results():
    """
    Display the Operator view results section with gauge and metrics.
    """

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
# PDF REPORT GENERATION
# =========================================================


def build_operator_pdf_report():
    """
    Build a clear PDF report for the current Operator view.
    The report includes the selected options and the result gauge.
    """

    if not st.session_state.get("operator_model_complete", False):
        return BytesIO().getvalue()

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
    return buffer
