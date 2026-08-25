Transient Stability Limit Prediction GUI

A Python-based data product developed for the DSC-580 Milestone 3 project. The application provides a graphical user interface (GUI) for exploring power-system data and evaluating machine-learning models designed to predict transient stability limits.

The project demonstrates how machine-learning models can be incorporated into a user-oriented data product for power-system analysis. Separate functionality is provided for operator and data scientist use cases so that information can be presented according to different user needs.

Project Purpose

Traditional transient stability analysis can require significant computational effort, particularly when many operating conditions and contingencies must be evaluated. This project explores the use of trained machine-learning regression models to estimate transient stability limits from power-system data.

The GUI provides a common interface through which users can:

Load and explore power-system data.
Review data used for transient stability analysis.
Generate machine-learning predictions.
Compare multiple trained machine-learning models.
Evaluate model accuracy and prediction speed.
Present analytical information differently depending on the user's role.
Repository Structure
Milestone-3-GUI/
│
├── GUI Rev 1.2.py
├── Operator.py
├── Data_Scientist.py
├── long_df_head.csv
├── requirements.txt
│
└── saved_models/
GUI Rev 1.2.py

Main application file containing the graphical user interface and coordinating the different application views.

Operator.py

Contains functionality intended for the operator view of the data product. This view is intended to emphasize operationally relevant results rather than detailed machine-learning model evaluation.

Data_Scientist.py

Contains functions used by the data scientist view, including benchmarking the trained machine-learning models.

The benchmarking function evaluates models using:

Coefficient of determination (R²)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Total prediction time
Average single-sample prediction time

The same randomly selected observations are used for each model during benchmarking to provide a consistent comparison between models.

saved_models/

Contains previously trained machine-learning models stored as Joblib files.

Models currently evaluated by the data scientist benchmarking functionality include:

Histogram Gradient Boosting
K-Nearest Neighbors (KNN)
Support Vector Regression with an RBF kernel (SVR-RBF)

Because the models are saved after training, the GUI can load them and perform predictions without retraining the models each time the application starts.

long_df_head.csv

Sample power-system dataset that can be used for demonstrating and testing application functionality.

requirements.txt

Lists the Python packages and versions required to run the application.
