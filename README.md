# Weather Prediction Using Machine Learning

## Overview
This project aims to predict the weather using machine learning models based on the Seattle weather dataset. The dataset contains daily observations of weather-related features such as precipitation, temperature, and wind speed. Three different models—Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor—are trained and optimized using GridSearchCV to achieve the best performance.

## Dataset
The dataset used in this project is `seattle-weather.csv`, which includes the following columns:
- **date:** Date of the observation.
- **precipitation:** Amount of precipitation in inches.
- **temp_max:** Maximum temperature recorded in Fahrenheit.
- **wind:** Wind speed in miles per hour.
- **weather:** Weather conditions (encoded as integers).

## Project Structure
- `weather_predictor.py`: Contains the code for model training, testing, and evaluation.
- `seattle-weather.csv`: The dataset file used for training and testing.
- `decesion_tree_model.pkl`, `random_forest_model.pkl`, `xgb_model.pkl`: Saved models after training and optimization.
- `README.md`: This documentation file.

## Dependencies
The following Python libraries are required:
- pandas
- scikit-learn
- xgboost
- matplotlib
- joblib

You can install these dependencies using:

pip install -r requirements.txt

## How to Run the Project
1. Clone the repository:

   git clone https://github.com/muonorb/Weather-Predictor.git
  
3. Navigate to the project directory:
  
   cd Weather-Predictor

4. Ensure the dataset file `seattle-weather.csv` is in the project directory.
5. Run the `weather_predictor.py` script:
   
   python weather_predictor.py
   
   
## Models Overview
Three machine learning models are implemented and evaluated in this project:

### 1. Decision Tree Regressor
- **Hyperparameters Tuned:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **GridSearchCV:** Used to find the best model configuration based on negative mean squared error.
- **Evaluation Metrics:** Mean Absolute Error (MAE), R² score.

### 2. Random Forest Regressor
- **Hyperparameters Tuned:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **GridSearchCV:** Used for hyperparameter optimization.
- **Evaluation Metrics:** Mean Absolute Error (MAE), R² score.

### 3. XGBoost Regressor
- **Hyperparameters Tuned:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- **GridSearchCV:** Used for hyperparameter tuning.
- **Evaluation Metrics:** Mean Absolute Error (MAE), R² score.

## Results
After training and optimizing each model, the performance is evaluated using MAE and R² score. Scatter plots are generated to visualize the model predictions against actual values and the model is considered.

## Conclusion
This project demonstrates the use of multiple regression algorithms to predict weather conditions based on historical weather data. The models are fine-tuned using GridSearchCV for optimal performance.

## License
This project is licensed under the MIT License 

