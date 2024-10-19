import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import datetime
from io import StringIO
import random
from geopy.distance import geodesic
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
ais_train['datetime'] = pd.to_datetime(ais_train['time']).dt.tz_localize(None)
ais_train['hour'] = ais_train['datetime'].dt.hour
ais_train['day'] = ais_train['datetime'].dt.day
ais_train['month'] = ais_train['datetime'].dt.month
ais_train['year'] = ais_train['datetime'].dt.year
ais_train['minute'] = ais_train['datetime'].dt.minute

ais_test = pd.read_csv('ais_test.csv', sep=',')
# Preparing the test data (ais_test.csv) for prediction
ais_test['datetime'] = pd.to_datetime(ais_test['time']).dt.tz_localize(None)
ais_test['hour'] = ais_test['datetime'].dt.hour
ais_test['day'] = ais_test['datetime'].dt.day
ais_test['month'] = ais_test['datetime'].dt.month
ais_test['year'] = ais_test['datetime'].dt.year
ais_test['minute'] = ais_test['datetime'].dt.minute

stm = open('schedules_to_may_2024.csv', 'r')
stm = ''.join([i for i in stm]).replace('"', '')
schedules = pd.read_csv(StringIO(stm), sep='|')
schedules['datetime'] = pd.to_datetime(schedules['arrivalDate']).dt.tz_localize(None)
schedules = schedules.dropna(subset=['datetime'])
schedules['hour'] = schedules['datetime'].dt.hour
schedules['day'] = schedules['datetime'].dt.day
schedules['month'] = schedules['datetime'].dt.month
schedules['year'] = schedules['datetime'].dt.year
schedules['minute'] = schedules['datetime'].dt.minute
#schedules = schedules.drop_duplicates(subset=['vesselId', 'day', 'month', 'year'], keep='last')
schedules = schedules.sort_values(by=['datetime','vesselId'])

vessels = pd.read_csv('vessels.csv', sep='|')

max_sog_per_vessel = ais_train[['vesselId', 'sog']].groupby('vesselId')['sog'].max().reset_index()
max_sog_per_vessel.to_csv('testing_max_sog.csv', sep=',', index=False)
vessels = pd.merge(vessels, max_sog_per_vessel, on='vesselId', how='left')


vessels['maxSpeedUpdated'] = np.where(vessels['maxSpeed'].isna(), vessels['sog'], vessels['maxSpeed'])
vessels['maxSpeedUpdated'] = np.where(vessels['maxSpeedUpdated'] > 50, None, vessels['maxSpeedUpdated'])
#ports = pd.read_csv('ports.csv')

ais_train = pd.merge(ais_train, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_train = pd.merge_asof(ais_train, schedules, by='vesselId', on='datetime', direction='forward', suffixes=('', '_sch'))
ais_test = pd.merge(ais_test, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_test = pd.merge_asof(ais_test, schedules, by='vesselId', on='datetime', direction='forward', suffixes=('', '_sch'))
# Preprocessing: convert 'datetime' column to datetime
print("sjekk3", len(ais_test['vesselId']))
#ais_test.head(1).to_csv('kollonnene.csv', sep=',', index=False)

vessel_encoder = LabelEncoder()
ais_train['vesselId'] = vessel_encoder.fit_transform(ais_train['vesselId'])

# Select features and target
features = ['vesselId', 'hour', 'day', 'month', 'minute', 'vesselType', 'yearBuilt', 'length', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude', 'portLongitude', 'hour_sch', 'day_sch', 'month_sch', 'minute_sch', 'maxSpeedUpdated']#'vesselType', 'yearBuilt', , 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute']#['cog', 'sog', 'rot', 'heading', 'navstat', 'latitude', 'longitude', 'vesselId', 'portId', 'hour', 'day', 'month', 'year', 'minute']
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
    print('median:', data['maxSpeedUpdated'].median())
    data['maxSpeedUpdated'].fillna(data['maxSpeedUpdated'].median(), inplace=True)
    data['portLatitude'].fillna(0, inplace=True)
    data['portLongitude'].fillna(0, inplace=True)
    data['hour_sch'].fillna(0, inplace=True)
    data['minute_sch'].fillna(0, inplace=True)
    data['day_sch'].fillna(0, inplace=True)
    data['month_sch'].fillna(0, inplace=True)
    return data

# Handle missing values (if any)
#ais_train = ais_train.dropna(subset=features + target) #ais_train.replace(to_replace='None', value=np.nan).dropna()
ais_train = feature_engineering(ais_train)
#ais_train[['minute']].to_csv('x_data_0.csv', sep=',', index=False)
#ais_train[['portLongitude']].to_csv('x_data_1.csv', sep=',', index=False)
#ais_train[['vesselId']].to_csv('x_data_2.csv', sep=',', index=False)
#ais_train[['breadth']].to_csv('x_data_3.csv', sep=',', index=False)
#ais_train[['GT']].to_csv('x_data_4.csv', sep=',', index=False)
#ais_train[['length']].to_csv('x_data_5.csv', sep=',', index=False)
#ais_train[['month']].to_csv('x_data_6.csv', sep=',', index=False)
#ais_train[['minute_sch']].to_csv('x_data_7.csv', sep=',', index=False)
#ais_train[['yearBuilt']].to_csv('x_data_8.csv', sep=',', index=False)
#ais_train[['hour_sch']].to_csv('x_data_9.csv', sep=',', index=False)
#ais_train[['hour']].to_csv('x_data_10.csv', sep=',', index=False)
#ais_train[['CEU']].to_csv('x_data_11.csv', sep=',', index=False)
#ais_train[['portLatitude']].to_csv('x_data_12.csv', sep=',', index=False)
#ais_train[['vesselType']].to_csv('x_data_13.csv', sep=',', index=False)
#ais_train[['year']].to_csv('x_data_14.csv', sep=',', index=False)
#ais_train[['day']].to_csv('x_data_15.csv', sep=',', index=False)
#ais_train[['DWT']].to_csv('x_data_16.csv', sep=',', index=False)
#ais_train[['latitude']].to_csv('y_data_1.csv', sep=',', index=False)
#ais_train[['longitude']].to_csv('y_data_2.csv', sep=',', index=False)
#ais_train[['maxSpeedUpdated']].to_csv('x_data_17.csv', sep=',', index=False)
#ais_train[['month_sch']].to_csv('x_data_18.csv', sep=',', index=False)
#ais_train[['day_sch']].to_csv('x_data_19.csv', sep=',', index=False)
#ais_train[['year_sch']].to_csv('x_data_20.csv', sep=',', index=False)
# Train-test split
X = ais_train[features]
X.tail(10000).to_excel('datasetcheck.xlsx')
y = ais_train[target]

model = RandomForestRegressor(n_estimators=1, random_state=rs)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForest model
model.fit(X_train_scaled, y_train)

# Predict on validation set
y_pred = model.predict(X_val_scaled)

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
#ais_test[['portLongitude', 'hour_sch', 'minute_sch']].to_csv('x_test_1.csv', sep=',', index=False)
#ais_test[['vesselId', 'hour', 'day', 'month', 'year']].to_csv('x_test_2.csv', sep=',', index=False)
#ais_test[['breadth', 'CEU', 'DWT', 'GT', 'portLatitude']].to_csv('x_test_3.csv', sep=',', index=False)
ais_test[['maxSpeedUpdated', 'month_sch', 'day_sch']].to_csv('x_test_4.csv', sep=',', index=False)
# Use the same features as training
X_test = ais_test[features]

# Scale the test set
X_test_scaled = scaler.transform(X_test)

# Predict on the test set
test_predictions = model.predict(X_test_scaled)

# Save the predictions
predictions_df = pd.DataFrame(test_predictions, columns=['longitude_predicted', 'latitude_predicted'])
#predictions_df['ID'] = ais_test['ID']
#predictions_df = predictions_df[-1:] + predictions_df[:-1]
#predictions_df = predictions_df.round(5)
predictions_df.to_csv('ais_test_predictions_1.csv', index=True)
print(datetime.datetime.today().strftime("%HH:%MM %dd"))