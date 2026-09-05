# -----------------------------------
# Import necessary libraries
# -----------------------------------

import streamlit as st
import pandas as pd

from textwrap import dedent

from operator_reporting import (
    operator_obtain_data,
    operator_scrub_data,
    operator_explore_data,
    operator_model_data,
    operator_interpret_results,
    run_operator_pipeline,
    reset_operator_pipeline,
    display_operator_pipeline_controls,
    display_operator_results,
)
from data_scientist_reporting import (
    ds_obtain_data,
    ds_scrub_data,
    ds_explore_data,
    ds_model_data,
    ds_interpret_results,
    run_ds_pipeline,
    reset_ds_pipeline,
    display_ds_pipeline_controls,
    display_ds_exploratory_results,
    display_ds_benchmark_results,
    display_ds_pdf_export,
)


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
    Revision 9/5/2026
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

    display_operator_results()

    st.divider()

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

    display_ds_pipeline_controls(DATA_PATH)

    st.divider()


    # -----------------------------------
    # Exploratory Results
    # -----------------------------------

    display_ds_exploratory_results()

    st.divider()


    # -----------------------------------
    # Benchmark Results
    # -----------------------------------

    display_ds_benchmark_results()

    st.divider()


    # -----------------------------------
    # PDF Export
    # -----------------------------------

    display_ds_pdf_export()

