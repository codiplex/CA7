import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load processed data
data = np.load('processed_data.npz')
features = data['X']
target = data['y']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Function to train, evaluate and log a model with MLflow
def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on validation set
        val_predictions = model.predict(X_val)
        
        # Calculate metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        # Log parameters and metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        return val_rmse, val_mae, val_r2

# Train and log Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
train_and_log_model(dt_model, "Decision_Tree", X_train, y_train, X_val, y_val)

# Train and log Random Forest
rf_model = RandomForestRegressor(random_state=42)
train_and_log_model(rf_model, "Random_Forest", X_train, y_train, X_val, y_val)

# Train and log Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
train_and_log_model(gb_model, "Gradient_Boosting", X_train, y_train, X_val, y_val)

# Train and log Support Vector Machine
svr_model = SVR()
train_and_log_model(svr_model, "Support_Vector_Machine", X_train, y_train, X_val, y_val)

# Train and log Neural Network
def build_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

nn_model = build_nn_model((X_train.shape[1],))
nn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=2)

with mlflow.start_run(run_name="Neural_Network"):
    # Log the neural network model
    mlflow.tensorflow.log_model(nn_model, "Neural_Network")

    # Calculate validation RMSE, MAE, and R-squared
    val_predictions = nn_model.predict(X_val).flatten()
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    # Log parameters and metrics
    mlflow.log_param("model", "Neural_Network")
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_r2", val_r2)
