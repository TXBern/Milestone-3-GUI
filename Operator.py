# -----------------------------------
# Import necessary libraries
# -----------------------------------

#import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------
# Page configuration
# -----------------------------------

#df = pd.read_csv("C:\\Users\\bernh\\Documents\\GCU\\DSC-580\\Milestone 3\\long_df_head.csv")

# -----------------------------------
# Loading the model with highest accuracy
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "HistGradientBoosting.joblib")

# -----------------------------------
# Select a Random Sample and Make a Prediction
# -----------------------------------

def get_random_prediction(df):
    """
    Select a random sample from df and use the saved
    Random Forest model to predict P_diff_target0.
    """
    # Select one random row
    random_row = df.sample(n=1)

    # Get current interface flow from the selected row
    current_flow = random_row["P_difference"].iloc[0]

    # Get the features expected by the model
    X_sample = random_row[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(X_sample)[0]

    return prediction, current_flow



# prediction, current_flow = get_random_prediction(df)

# print(f"Current Flow: {current_flow:,.2f} MW")
# print(f"Predicted Stability Limit: {prediction:,.2f} MW")
