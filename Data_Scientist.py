# -----------------------------------
# Import necessary libraries
# -----------------------------------

import joblib
import pandas as pd
import numpy as np
import time
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# -----------------------------------
# Loading the models to be compared
# -----------------------------------

models = {
    "Random Forest": joblib.load("C:\\Users\\bernh\\Documents\\GCU\\DSC-580\\Milestone 3\\saved_models\\RandomForest.joblib"),
    "HistGradientBoosting": joblib.load("C:\\Users\\bernh\\Documents\\GCU\\DSC-580\\Milestone 3\\saved_models\\HistGradientBoosting.joblib"),
    "KNN": joblib.load("C:\\Users\\bernh\\Documents\\GCU\\DSC-580\\Milestone 3\\saved_models\\KNN.joblib"),
    "SVR_RBF": joblib.load("C:\\Users\\bernh\\Documents\\GCU\\DSC-580\\Milestone 3\\saved_models\\SVR_RBF.joblib")
}

# -----------------------------------
# Select a Random Sample and Compare Model Performance
# -----------------------------------
def benchmark_models(
    df,
    target_column="P_diff_target0_est",
    sample_size=1000
):
    """
    Randomly select samples and benchmark all saved models.

    Returns:
        pandas DataFrame containing R2, RMSE, MAE,
        total prediction time, and average prediction time.
    """

    # Don't request more rows than are available
    sample_size = min(sample_size, len(df))

    # Select SAME random samples for every model
    sample_df = df.sample(
        n=sample_size,
        random_state=42
    )

    # Actual target values
    y_actual = sample_df[target_column]

    results = []

    # -----------------------------------
    # Test each model
    # -----------------------------------

    for model_name, model in models.items():

        # Select exactly the features expected by this model
        X_sample = sample_df[model.feature_names_in_]

        # Start timer
        start_time = time.perf_counter()

        # Predict all samples
        predictions = model.predict(X_sample)

        # Stop timer
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # Average prediction time for ONE sample
        avg_prediction_time = total_time / sample_size

        # Accuracy metrics
        r2 = r2_score(
            y_actual,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_actual,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_actual,
            predictions
        )

        single_sample = X_sample.iloc[[0]]

        single_times = []

        for _ in range(100):

            start = time.perf_counter()

            model.predict(single_sample)

            end = time.perf_counter()

            single_times.append(end - start)

        avg_single_prediction_time = (
            np.mean(single_times) * 1000
        )

        results.append({
            "Model": model_name,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "Total Time (sec)": total_time,
            "Single Prediction Time (ms)": (
                avg_single_prediction_time
            )
        })

    results_df = pd.DataFrame(results)

    return results_df