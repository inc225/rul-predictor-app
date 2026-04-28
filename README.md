# Predictive Maintenance for Aircraft Engines: A Machine Learning Approach to RUL Prediction

## Project Description
This project predicts the Remaining Useful Life (RUL) of aircraft turbofan engines using supervised machine learning. The goal is to support predictive maintenance by estimating how many operating cycles remain before an engine is likely to fail. Accurate RUL estimates can help maintenance teams reduce unplanned downtime, improve safety, and avoid replacing parts too early.

## Dataset
The project uses the FD001 subset of the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.

Files used:
- `data/train_FD001.txt`: complete run-to-failure trajectories for 100 training engines
- `data/test_FD001.txt`: partial trajectories for 100 test engines
- `data/RUL_FD001.txt`: true RUL labels for the final observed test cycle of each test engine

Each row contains:
- engine unit ID
- operating cycle
- 3 operational settings
- 21 sensor readings

The target variable for training is:

`RUL = max_cycle_for_engine - current_cycle`

A capped RUL target of 130 cycles is used to reduce the influence of very early-life observations where degradation is not strongly visible.

## Data Source
NASA Prognostics Center of Excellence, C-MAPSS Turbofan Engine Degradation Simulation Data. The dataset is commonly cited from:

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. International Conference on Prognostics and Health Management.

## Methods
The workflow includes:
1. Load and clean FD001 files.
2. Assign column names for engine ID, cycle, settings, and sensors.
3. Generate RUL labels for training trajectories.
4. Remove constant or low-information features.
5. Train and compare three regression models:
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Evaluate predictions on the final cycle of each test engine using the official RUL labels.
7. Save the best model for Streamlit deployment.

## Final Model Results
Evaluation uses capped RUL labels with a cap of 130 cycles.

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Random Forest | 18.20 | 13.18 | 0.800 |
| Gradient Boosting | 18.45 | 12.96 | 0.794 |
| Ridge Regression | 21.23 | 16.93 | 0.728 |

The selected model is **Random Forest**, because it achieved the lowest RMSE on the FD001 test set.

## Repository Structure
```text
LnameFname/
├── app.py
├── aux_1.py
├── main.py
├── README.md
├── ReadMe.txt
├── requirements.txt
├── data/
│   ├── readme_data.txt
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── figures/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_comparison.png
├── models/
│   └── best_rul_model.joblib
└── reports/
    ├── feature_importance.csv
    ├── model_results.csv
    └── test_predictions.csv
```

## Required Packages
Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit

## How to Train and Evaluate
From the project folder, run:

```bash
python main.py
```

This trains the models, evaluates them, saves the best model to `models/best_rul_model.joblib`, and writes result files to the `reports/` and `figures/` folders.

## How to Run the Web App Locally
From the project folder, run:

```bash
streamlit run app.py
```

The app supports:
- uploading engine sensor data in CSV format
- entering one engine cycle manually
- returning the predicted RUL in operating cycles

## Streamlit Cloud Deployment Instructions
1. Push this project folder to a public GitHub repository.
2. Go to Streamlit Community Cloud.
3. Choose the repository and select `app.py` as the entry point.
4. Confirm that `requirements.txt` is included.
5. Deploy and paste the app URL into the submission form.

## Notes
- The best saved model is included so the Streamlit app can run without retraining.
- The data files are included in the `data/` folder for reproducibility.
- If your instructor requires no raw data in the ZIP, keep `data/readme_data.txt` and remove the three `.txt` data files before uploading.
