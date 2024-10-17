import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import datetime
from io import StringIO
import random
from sklearn.model_selection import GridSearchCV
#random.seed(420)

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
#ports = pd.read_csv('ports.csv')
ais_test = pd.merge(ais_test, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_test = pd.merge(ais_test, schedules, left_on=['vesselId', 'day', 'month', 'year'], right_on=['vesselId', 'day', 'month', 'year'], how='left')
ais_train = pd.merge(ais_train, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_train = pd.merge(ais_train, schedules, left_on=['vesselId', 'day', 'month', 'year'], right_on=['vesselId', 'day', 'month', 'year'], how='left')
# Preprocessing: convert 'time' column to datetime


vessel_encoder = LabelEncoder()
ais_train['vesselId'] = vessel_encoder.fit_transform(ais_train['vesselId'])

# Select features and target
features = ['vesselId', 'hour', 'day', 'month', 'year', 'minute', 'yearBuilt', 'length', 'vesselType', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute']#['cog', 'sog', 'rot', 'heading', 'navstat', 'latitude', 'longitude', 'vesselId', 'portId', 'hour', 'day', 'month', 'year', 'minute']
target = ['latitude', 'longitude']  # Predicting next position (latitude, longitude)

def feature_engineering(data):

    data['vesselType'].fillna(83, inplace=True) #data['vesselType'].mode() -> dette er 83 men funker ikke??
    data['yearBuilt'].fillna(data['yearBuilt'].median(), inplace=True)
    data['length'].fillna(data['length'].median(), inplace=True)
    data['breadth'].fillna(data['breadth'].median(), inplace=True)
    data['CEU'].fillna(data['CEU'].median(), inplace=True)
    data['DWT'].fillna(data['DWT'].median(), inplace=True)
    data['GT'].fillna(data['GT'].median(), inplace=True)
    data['portLatitude'].fillna(0, inplace=True)
    data['portLongitude'].fillna(0, inplace=True)
    data['sch_hour'].fillna(0, inplace=True)
    data['sch_minute'].fillna(0, inplace=True)
    return data

# Handle missing values (if any)
#ais_train = ais_train.dropna(subset=features + target) #ais_train.replace(to_replace='None', value=np.nan).dropna()
ais_train = feature_engineering(ais_train)


# Train-test split
X = ais_train[features]
#print(X.tail(10000))
y = ais_train[target]

param_grid = {'n_neighbors': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}


#model = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')#, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)#, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForest model
model.fit(X_train_scaled, y_train)

# Predict on validation set
y_pred = model.predict(X_val_scaled)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Absolute Error: {mse}')
print(f'Root Mean Squared Error: {np.sqrt(mse)}')
print(f'R2-score: {r2_score(y_val, y_pred)}')
print(f'Variance Score: {explained_variance_score(y_val, y_pred)}')




ais_test['vesselId'] = vessel_encoder.transform(ais_test['vesselId'])

ais_test = feature_engineering(ais_test)

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
predictions_df.to_csv('ais_test_predictions_5.csv', index=True)
print(datetime.datetime.today().strftime("%HH:%MM %dd"))