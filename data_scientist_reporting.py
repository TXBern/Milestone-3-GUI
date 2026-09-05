# -----------------------------------
# Data Scientist View Reporting Module
# -----------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import matplotlib.pyplot as plt
from textwrap import dedent

from Data_Scientist import benchmark_models


# =========================================================
# DATA SCIENTIST OSEMN FUNCTIONS
# =========================================================


def ds_obtain_data(DATA_PATH):
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
        # ds_obtain_data should have been called already
        return

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


def run_ds_pipeline(DATA_PATH):
    """
    Run the complete Data Scientist OSEMN pipeline.
    """

    ds_obtain_data(DATA_PATH)

    ds_scrub_data()

    ds_explore_data()

    ds_model_data()

    ds_interpret_results()


# =========================================================
# PIPELINE RESET FUNCTIONS
# =========================================================


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
# DATA SCIENTIST PIPELINE CONTROLS
# =========================================================


def display_ds_pipeline_controls(DATA_PATH):

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

                run_ds_pipeline(DATA_PATH)

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

            ds_obtain_data(DATA_PATH)

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
# DATA SCIENTIST RESULTS DISPLAY
# =========================================================


def display_ds_exploratory_results():
    """
    Display the exploratory data analysis results.
    """

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


def display_ds_benchmark_results():
    """
    Display the model benchmark results with scatter plot and metrics table.
    """

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


# =========================================================
# PDF REPORT GENERATION
# =========================================================


def build_ds_pdf_report():
    """
    Build a PDF report for the Data Scientist view.
    The report includes summary statistics, model benchmarks, and interpretation results.
    """

    if not st.session_state.get("ds_model_complete", False):
        return BytesIO()

    data_source = st.session_state.get("ds_data_source", "Read File")
    scrub_method = st.session_state.get("ds_scrub_method", "Default")
    
    results = st.session_state.get("benchmark_results", pd.DataFrame())
    best_accuracy_model = st.session_state.get("ds_best_accuracy_model", "N/A")
    fastest_model = st.session_state.get("ds_fastest_model", "N/A")
    summary_stats = st.session_state.get("ds_summary_stats", pd.DataFrame())

    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        # ===================================================
        # PAGE 1 - TITLE AND CONFIGURATION
        # ===================================================
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Data Scientist View Report",
            fontsize=22,
            fontweight="bold",
            y=0.97,
        )

        config_rows = [
            ("Report Type", "Model Development & Benchmarking"),
            ("Data Source", data_source),
            ("Scrubbing Method", scrub_method),
            ("", ""),  # blank row
            ("Model Performance Summary", ""),
            ("Best Accuracy Model (R²)", best_accuracy_model),
            ("Fastest Prediction Model", fastest_model),
        ]

        summary_x = 0.10
        summary_y = 0.85
        for label, value in config_rows:
            if label == "Model Performance Summary":
                fig.text(
                    summary_x,
                    summary_y,
                    label,
                    fontsize=12,
                    va="center",
                    fontweight="bold",
                )
            else:
                fig.text(
                    summary_x,
                    summary_y,
                    f"{label}: {value}",
                    fontsize=11,
                    va="center",
                )
            summary_y -= 0.05

        # Add dataset info
        info_y = summary_y - 0.05
        fig.text(
            summary_x,
            info_y,
            "Dataset Information",
            fontsize=12,
            va="center",
            fontweight="bold",
        )
        info_y -= 0.05

        if hasattr(st.session_state, "ds_clean_df") and st.session_state.ds_clean_df is not None:
            df_info = st.session_state.ds_clean_df
            fig.text(
                summary_x,
                info_y,
                f"Samples: {len(df_info)}",
                fontsize=10,
                va="center",
            )
            info_y -= 0.04
            fig.text(
                summary_x,
                info_y,
                f"Features: {len(df_info.columns)}",
                fontsize=10,
                va="center",
            )

        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================
        # PAGE 2 - BENCHMARK RESULTS TABLE
        # ===================================================
        if not results.empty:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            fig.suptitle("Model Benchmark Results", fontsize=16, fontweight="bold", y=0.98)

            # Create table with benchmark results
            table_data = [["Model", "R²", "RMSE", "MAE", "Prediction Time (ms)"]]
            
            for idx, row in results.iterrows():
                table_data.append([
                    str(row.get("Model", "N/A")),
                    f"{row.get('R²', 0):.4f}",
                    f"{row.get('RMSE', 0):.2f}",
                    f"{row.get('MAE', 0):.2f}",
                    f"{row.get('Single Prediction Time (ms)', 0):.4f}",
                ])

            table = ax.table(
                cellText=table_data,
                colLabels=None,
                cellLoc="center",
                loc="center",
                bbox=[0.05, 0.1, 0.9, 0.85],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)

            # Style header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor("#4472C4")
                table[(0, i)].set_text_props(weight="bold", color="white")

            pdf.savefig(fig)
            plt.close(fig)

        # ===================================================
        # PAGE 3 - SUMMARY STATISTICS
        # ===================================================
        if not summary_stats.empty:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            fig.suptitle("Summary Statistics", fontsize=16, fontweight="bold", y=0.98)

            # Create table with summary statistics
            summary_table_data = [["Statistic"] + list(summary_stats.columns[:6])]
            
            for idx_name in summary_stats.index[:8]:  # Limit to key stats
                row = [str(idx_name)]
                for col in summary_stats.columns[:6]:
                    val = summary_stats.loc[idx_name, col]
                    row.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                summary_table_data.append(row)

            summary_table = ax.table(
                cellText=summary_table_data,
                colLabels=None,
                cellLoc="center",
                loc="center",
                bbox=[0.05, 0.1, 0.9, 0.85],
            )
            summary_table.auto_set_font_size(False)
            summary_table.set_fontsize(8)
            summary_table.scale(1, 1.5)

            # Style header row
            for i in range(len(summary_table_data[0])):
                summary_table[(0, i)].set_facecolor("#70AD47")
                summary_table[(0, i)].set_text_props(weight="bold", color="white")

            pdf.savefig(fig)
            plt.close(fig)

    buffer.seek(0)
    return buffer


def display_ds_pdf_export():
    """
    Display PDF export section in an expander.
    """

    if st.session_state.ds_model_complete:

        with st.expander(
            "Export Analysis Results as PDF"
        ):

            pdf_buffer = build_ds_pdf_report()

            st.download_button(
                "📥 Download PDF Report",
                pdf_buffer,
                "data_scientist_analysis_report.pdf",
                "application/pdf",
                use_container_width=True,
            )

    else:

        st.info(
            "Run the Data Scientist Model stage or the "
            "entire Data Scientist pipeline to generate "
            "a PDF report."
        )
