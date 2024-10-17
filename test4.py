import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import datetime
from io import StringIO
import random
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

rs = 42
random.seed(rs)

def calculate_weighted_geodetic_error(true_df, pred_df, weights):
    """
    Calculate the weighted average of the mean geodetic distances (in kilometers).
    
    Args:
    - true_df: DataFrame containing true 'latitude' and 'longitude' columns.
    - pred_df: DataFrame containing predicted 'latitude' and 'longitude' columns.
    - weights: Series or DataFrame column containing the weights to apply to each geodetic distance.
    
    Returns:
    - weighted_avg_distance: Weighted average geodetic distance.
    """
    
    # Ensure the inputs have matching lengths
    assert len(true_df) == len(pred_df) == len(weights), "The lengths of inputs must match."
    
    # Initialize total weighted sum and total weight
    weighted_distance_sum = 0.0
    total_weight = 0.0
    
    # Loop over each row and calculate geodetic distance
    for i, (true_row, pred_row, weight) in enumerate(zip(true_df.itertuples(index=False), pred_df.itertuples(index=False), weights)):
        # Ensure latitude and longitude are valid numeric types
        true_pos = (float(true_row.latitude), float(true_row.longitude))
        pred_pos = (float(pred_row.latitude), float(pred_row.longitude))
        
        # Calculate geodetic distance in kilometers
        distance_km = geodesic(true_pos, pred_pos).kilometers
        
        # Accumulate weighted distance and total weight
        weighted_distance_sum += distance_km * weight
        total_weight += weight
    
    # Calculate the weighted average geodetic distance
    weighted_avg_distance = weighted_distance_sum / total_weight if total_weight != 0 else 0
    
    return weighted_avg_distance


print(datetime.datetime.today().strftime("%HH:%MM %dd"))
# Load datasets
ais_train = pd.read_csv('ais_train.csv', sep='|')
# Feature Engineering: Extract relevant features from time
ais_train['time'] = pd.to_datetime(ais_train['time'])
ais_train['hour'] = ais_train['time'].dt.hour
ais_train['day'] = ais_train['time'].dt.day
ais_train['month'] = ais_train['time'].dt.month
ais_train['year'] = ais_train['time'].dt.year
ais_train['minute'] = ais_train['time'].dt.minute

ais_test = pd.read_csv('ais_test.csv', sep=',')
# Preparing the test data (ais_test.csv) for prediction
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_test['hour'] = ais_test['time'].dt.hour
ais_test['day'] = ais_test['time'].dt.day
ais_test['month'] = ais_test['time'].dt.month
ais_test['year'] = ais_test['time'].dt.year
ais_test['minute'] = ais_test['time'].dt.minute

stm = open('schedules_to_may_2024.csv', 'r')
stm = ''.join([i for i in stm]).replace('"', '')
schedules = pd.read_csv(StringIO(stm), sep='|')
schedules['arrivalDate'] = pd.to_datetime(schedules['arrivalDate'])
schedules['sch_hour'] = schedules['arrivalDate'].dt.hour
schedules['day'] = schedules['arrivalDate'].dt.day
schedules['month'] = schedules['arrivalDate'].dt.month
schedules['year'] = schedules['arrivalDate'].dt.year
schedules['sch_minute'] = schedules['arrivalDate'].dt.minute
schedules = schedules.drop_duplicates(subset=['vesselId', 'day', 'month', 'year'], keep='last')

vessels = pd.read_csv('vessels.csv', sep='|')

max_sog_per_vessel = ais_train[['vesselId', 'sog']].groupby('vesselId')['sog'].max().reset_index()
vessels = pd.merge(vessels, max_sog_per_vessel, on='vesselId', how='left')

vessels['maxSpeedUpdated'] = np.where(vessels['maxSpeed'].isna() | (vessels['sog'] == 0),  # If maxSpeed is NaN and sog > 0
    vessels['sog'],  # Use max_sog
    vessels['maxSpeed']  # Otherwise keep the existing maxSpeed
)
#ports = pd.read_csv('ports.csv')
ais_test = pd.merge(ais_test, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_test = pd.merge(ais_test, schedules, left_on=['vesselId', 'day', 'month', 'year'], right_on=['vesselId', 'day', 'month', 'year'], how='left')
ais_train = pd.merge(ais_train, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_train = pd.merge(ais_train, schedules, left_on=['vesselId', 'day', 'month', 'year'], right_on=['vesselId', 'day', 'month', 'year'], how='left')
# Preprocessing: convert 'time' column to datetime
print("sjekk3", len(ais_test['vesselId']))

vessel_encoder = LabelEncoder()
ais_train['vesselId'] = vessel_encoder.fit_transform(ais_train['vesselId'])

# Select features and target
features = ['vesselId', 'hour', 'day', 'month', 'minute', 'vesselType', 'yearBuilt', 'length', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute', 'maxSpeedUpdated']#'vesselType', 'yearBuilt', , 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute']#['cog', 'sog', 'rot', 'heading', 'navstat', 'latitude', 'longitude', 'vesselId', 'portId', 'hour', 'day', 'month', 'year', 'minute']
target = ['longitude', 'latitude']  # Predicting next position (latitude, longitude)

def feature_engineering(data):
    data['vesselType'].fillna(83, inplace=True) #data['vesselType'].mode() -> dette er 83 men funker ikke??
    data['yearBuilt'].fillna(data['yearBuilt'].median(), inplace=True)
    data['length'].fillna(data['length'].median(), inplace=True)
    data['breadth'].fillna(data['breadth'].median(), inplace=True)
    data['CEU'].fillna(data['CEU'].median(), inplace=True)
    data['DWT'].fillna(data['DWT'].median(), inplace=True)
    data['GT'].fillna(data['GT'].median(), inplace=True)
    data['maxSpeedUpdated'].replace(0, None)
    data['maxSpeedUpdated'].fillna(data['maxSpeedUpdated'].median(), inplace=True)
    data['portLatitude'].fillna(0, inplace=True)
    data['portLongitude'].fillna(0, inplace=True)
    data['sch_hour'].fillna(0, inplace=True)
    data['sch_minute'].fillna(0, inplace=True)
    return data

# Handle missing values (if any)
#ais_train = ais_train.dropna(subset=features + target) #ais_train.replace(to_replace='None', value=np.nan).dropna()
ais_train = feature_engineering(ais_train)

class CustomStackingRegressor():
    def __init__(self, estimators):
        self.stacking_regressor = StackingRegressor(estimators=estimators)

    def fit(self, X, y):
        self.stacking_regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_regressor.predict(X)

X = ais_train[features]
#print(X.tail(10000))
y = ais_train[target]

#model = ExtraTreesRegressor(n_estimators=120, random_state=rs)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForest model
estimators = [('rf', RandomForestRegressor(n_estimators=50)), ('et', ExtraTreesRegressor(n_estimators=50))]

stacked_models = []
for i in range(y_train.shape[1]):
    # Stacking Regressor for each target variable
    stacked_model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor(n_estimators=5000))
    stacked_model.fit(X_train, y_train.iloc[:, i])
    stacked_models.append(stacked_model)

# Predict on validation data for each target variable
y_pred = np.column_stack([model.predict(X_val) for model in stacked_models])

weights = []
for i in range(len(y_val)):
    weights.append(0.25)
# Evaluate the model
y_val_df = pd.DataFrame(y_val, columns=['longitude', 'latitude'])
y_pred_df = pd.DataFrame(y_pred, columns=['longitude', 'latitude'])
geodist = calculate_weighted_geodetic_error(y_val_df, y_pred_df, weights)
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Absolute Error: {mse}')
print(f'Root Mean Squared Error: {np.sqrt(mse)}')
print(f'R2-score: {r2_score(y_val, y_pred)}')
print(f'Variance Score: {explained_variance_score(y_val, y_pred)}')
print(f'Geo-distance: {geodist}')
print("")

ais_test['vesselId'] = vessel_encoder.transform(ais_test['vesselId'])

ais_test = feature_engineering(ais_test)

#ais_test[['minute', 'yearBuilt', 'length', 'vesselType']].to_csv('x_test.csv', sep=',', index=False)
#ais_test[['portLongitude', 'sch_hour', 'sch_minute']].to_csv('x_test_1.csv', sep=',', index=False)
#ais_test[['vesselId', 'hour', 'day', 'month', 'year']].to_csv('x_test_2.csv', sep=',', index=False)
#ais_test[['breadth', 'CEU', 'DWT', 'GT', 'portLatitude']].to_csv('x_test_3.csv', sep=',', index=False)
# Use the same features as training
X_test = ais_test[features]

# Scale the test set
X_test_scaled = scaler.transform(X_test)

# Predict on the test set
test_predictions = np.column_stack([model.predict(X_test_scaled) for model in stacked_models])

# Save the predictions
predictions_df = pd.DataFrame(test_predictions, columns=['longitude_predicted', 'latitude_predicted'])
#predictions_df['ID'] = ais_test['ID']
#predictions_df = predictions_df[-1:] + predictions_df[:-1]
#predictions_df = predictions_df.round(5)
predictions_df.to_csv('ais_test_predictions_4.csv', index=True)
print(datetime.datetime.today().strftime("%HH:%MM %dd"))