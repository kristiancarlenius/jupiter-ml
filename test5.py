import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import datetime
from io import StringIO
import random
from sklearn.model_selection import GridSearchCV
rs = 420
random.seed(rs)

print(datetime.datetime.today().strftime("%HH:%MM %dd"))

def fixcog(row):
    if(float(row['cog'])>=360 or row['cog']==None):
        return 0
    else:
        return row['cog']
        
def fixhead(row):
    if(float(row['heading'])>=360 or row['heading']==None):
        return 0
    else:
        return row['heading']
    
# Load datasets
ais_train = pd.read_csv('ais_train.csv', sep='|')
# Feature Engineering: Extract relevant features from time
ais_train['etaRaw'].fillna("00-00  00:00", inplace=True)
ais_train['month_eta'] = ais_train['etaRaw'].str.split('[:\s-]+').str[0].astype(int)
ais_train['day_eta'] = ais_train['etaRaw'].str.split('[:\s-]+').str[1].astype(int)
ais_train['hour_eta'] = ais_train['etaRaw'].str.split('[:\s-]+').str[2].astype(int)
ais_train['minute_eta'] = ais_train['etaRaw'].str.split('[:\s-]+').str[3].astype(int)

ais_train['month_eta'].replace(0, np.nan)
ais_train['day_eta'].replace(0, np.nan)


ais_train['datetime'] = pd.to_datetime(ais_train['time']).dt.tz_localize(None)
ais_train['time_diff'] = ais_train.groupby('vesselId')['datetime'].diff().dt.total_seconds()
ais_train['hour'] = ais_train['datetime'].dt.hour
ais_train['day'] = ais_train['datetime'].dt.day
ais_train['month'] = ais_train['datetime'].dt.month
ais_train['year'] = ais_train['datetime'].dt.year
ais_train['minute'] = ais_train['datetime'].dt.minute

ais_test = pd.read_csv('ais_test.csv', sep=',')
# Preparing the test data (ais_test.csv) for prediction
ais_test['datetime'] = pd.to_datetime(ais_test['time']).dt.tz_localize(None)
ais_test['time_diff'] = ais_test.groupby('vesselId')['datetime'].diff().dt.total_seconds()
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

#max_sog_per_vessel = ais_train[['vesselId', 'sog']].groupby('vesselId')['sog'].max().reset_index()
#max_sog_per_vessel.to_csv('testing_max_sog.csv', sep=',', index=False)
#vessels = pd.merge(vessels, max_sog_per_vessel, on='vesselId', how='left')


#vessels['maxSpeedUpdated'] = np.where(vessels['maxSpeed'].isna(), vessels['sog'], vessels['maxSpeed'])
#vessels['maxSpeedUpdated'] = np.where(vessels['maxSpeedUpdated'] > 50, None, vessels['maxSpeedUpdated'])

ports = pd.read_csv('ports.csv', sep='|')
ports.rename(columns = {'latitude':'port_latitude', 'longitude':'port_longitude'}, inplace=True)
print("port:", ports.columns)

liste = []
timesteps = 6
for i in range(1, timesteps):  # Only 3 previous time steps
    ais_train[f'lat_shift_{i}'] = ais_train.groupby('vesselId')['latitude'].shift(i)
    ais_train[f'lon_shift_{i}'] = ais_train.groupby('vesselId')['longitude'].shift(i)
    ais_train[f'minute_shift_{i}'] = ais_train.groupby('vesselId')['minute'].shift(i)
    ais_train[f'hour_shift_{i}'] = ais_train.groupby('vesselId')['hour'].shift(i)
    ais_train[f'day_shift_{i}'] = ais_train.groupby('vesselId')['day'].shift(i)
    ais_train[f'month_shift_{i}'] = ais_train.groupby('vesselId')['month'].shift(i)
    ais_train[f'cog_{i}'] = ais_train.groupby('vesselId')['cog'].shift(i)
    ais_train[f'heading_{i}'] = ais_train.groupby('vesselId')['heading'].shift(i)
    ais_train[f'lat_shift_{i}'].fillna(0, inplace=True)
    ais_train[f'lon_shift_{i}'].fillna(0, inplace=True)
    ais_train[f'minute_shift_{i}'].fillna(0, inplace=True)
    ais_train[f'hour_shift_{i}'].fillna(0, inplace=True)
    ais_train[f'day_shift_{i}'].fillna(0, inplace=True)
    ais_train[f'month_shift_{i}'].fillna(0, inplace=True) 
    ais_train[f'cog_{i}'] = ais_train.apply(fixcog, axis=1)
    ais_train[f'heading_{i}'] = ais_train.apply(fixhead, axis=1) 
    liste.append(f'lat_shift_{i}')
    liste.append(f'lon_shift_{i}')
    liste.append(f'minute_shift_{i}')
    liste.append(f'hour_shift_{i}')
    liste.append(f'day_shift_{i}')
    liste.append(f'month_shift_{i}')
    liste.append(f'cog_{i}')
    liste.append(f'heading_{i}')
liste.append('cog')
liste.append('heading')

for i in range(1, timesteps):  # Only 3 previous time steps
    ais_train[f'time_diff_{i}'] = ais_train.groupby('vesselId')['time_diff'].shift(i)
    ais_train[f'time_diff_{i}'].fillna(0, inplace=True)
    liste.append(f'time_diff_{i}')

ais_train = pd.merge(ais_train, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_train = pd.merge(ais_train, ports, left_on='portId', right_on='portId', how='left')
ais_train = pd.merge_asof(ais_train, schedules, by='vesselId', on='datetime', direction='forward', suffixes=('', '_sch'))

#ais_test = pd.merge(ais_test, vessels, left_on='vesselId', right_on='vesselId', how='left')
ais_test = pd.merge_asof(ais_test, schedules, by='vesselId', on='datetime', direction='forward', suffixes=('', '_sch'))

# port og schedual,  fuck schedual (only latetude and longetude), maxSpeed lite nyttig
#vi har en til form for datetime i schedual som vi kan bruke til å fylle, nå fyller vi schedual med port men er det bedre med fylling av port med schedual?
#######
#last known possision, how long ago,
#print("columns", ais_train.columns)
features = ['vesselId', 'hour', 'day', 'month', 'minute', 'vesselType', 'yearBuilt', 'length', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude', 'portLongitude', 'hour_sch', 'day_sch', 'month_sch', 'minute_sch'] + liste#'vesselType', 'yearBuilt', , 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute']#['cog', 'sog', 'rot', 'heading', 'navstat', 'latitude', 'longitude', 'vesselId', 'portId', 'hour', 'day', 'month', 'year', 'minute']
print("features:", features)
target = ['longitude', 'latitude']  # Predicting next position (latitude, longitude)


        
def feature_engineering(data):
    data['vesselType'].fillna(83, inplace=True) #data['vesselType'].mode() -> dette er 83 men funker ikke??
    data['yearBuilt'].fillna(data['yearBuilt'].median(), inplace=True)
    data['length'].fillna(data['length'].median(), inplace=True)
    data['breadth'].fillna(data['breadth'].median(), inplace=True)
    data['CEU'].fillna(data['CEU'].median(), inplace=True)
    data['DWT'].fillna(data['DWT'].median(), inplace=True)
    data['GT'].fillna(data['GT'].median(), inplace=True)

    data['portLatitude'].fillna(data['port_latitude'], inplace=True)
    data['portLongitude'].fillna(data['port_longitude'], inplace=True)
    data['portLatitude'].fillna(0, inplace=True)
    data['portLongitude'].fillna(0, inplace=True)

    data['hour_sch'].fillna(data['hour_eta'], inplace=True)
    data['minute_sch'].fillna(data['minute_eta'], inplace=True)
    data['day_sch'].fillna(data['day_eta'], inplace=True)
    data['month_sch'].fillna(data['month_eta'], inplace=True)

    data['hour_sch'].fillna(0, inplace=True)
    data['minute_sch'].fillna(0, inplace=True)
    data['day_sch'].fillna(data['day_eta'].median(), inplace=True)
    data['month_sch'].fillna(round(data['month_eta'].median()), inplace=True)
        
    data['cog'] = data.apply(fixcog, axis=1)
    data['heading'] = data.apply(fixhead, axis=1)

    data['time_diff'].fillna(0, inplace=True)

    return data

last_values = feature_engineering(ais_train).groupby('vesselId').last().reset_index()
missing_columns = [col for col in last_values.columns if col not in ais_test.columns and col != 'vesselId']
ais_test = ais_test.merge(last_values[['vesselId'] + missing_columns], on='vesselId', how='left')
test_merge = {}

for i in range(1, timesteps):  # Only 3 previous time steps
    test_merge[f'time_diff_{i}'] = ais_test.groupby('vesselId')['time_diff'].shift(i)
    ais_test[f'time_diff_{i}'] = np.where(test_merge[f'time_diff_{i}'].isna(), ais_test[f'time_diff_{i}'], test_merge[f'time_diff_{i}'])
    #test_merge[f'time_diff_{i}'].fillna(0, inplace=True)
    #liste.append(f'time_diff_{i}')

print("train:", len(ais_train['vesselId']))
print("test:", len(ais_test['vesselId']))
vessel_encoder = LabelEncoder()
ais_train['vesselId'] = vessel_encoder.fit_transform(ais_train['vesselId'])


# Handle missing values (if any)
#ais_train = ais_train.dropna(subset=features + target) #ais_train.replace(to_replace='None', value=np.nan).dropna()
ais_train = feature_engineering(ais_train)

count = 0
for i in features:
    ais_train[[i]].to_csv('x_data_'+str(count)+'.csv', sep=',', index=False)
    count += 1
#ais_train['longitude'].to_csv('y_data_2.csv', sep=',', index=False)
#ais_train['latitude'].to_csv('y_data_1.csv', sep=',', index=False)

# Train-test split
X = ais_train[features]
#print(X.tail(10000))
y = ais_train[target]


#TODO: create a second model, this one predicts the cog and heading, then after the prediction on the ais_test with both models, use it to update the shifted values to create a new prediction using the original model

#param_grid = {'n_neighbors': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
#model = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')#, random_state=42)
model = ExtraTreesRegressor(n_estimators=1, random_state=rs)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)

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

 
ais_test[['vesselId', 'hour', 'day', 'month', 'minute', 'vesselType', 'yearBuilt', 'length', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude']].to_csv('x_test.csv', sep=',', index=False)
ais_test[['portLongitude', 'hour_sch', 'day_sch', 'month_sch', 'minute_sch', 'lat_shift_1', 'lon_shift_1', 'minute_shift_1', 'hour_shift_1','day_shift_1', 'month_shift_1', 'cog_1', 'heading_1']].to_csv('x_test_1.csv', sep=',', index=False)
ais_test[['lat_shift_2', 'lon_shift_2', 'minute_shift_2', 'hour_shift_2', 'day_shift_2', 'month_shift_2', 'cog_2', 'heading_2', 'lat_shift_3', 'lon_shift_3', 'minute_shift_3', 'hour_shift_3']].to_csv('x_test_2.csv', sep=',', index=False)
ais_test[['day_shift_3', 'month_shift_3', 'cog_3', 'heading_3', 'lat_shift_4', 'lon_shift_4', 'minute_shift_4', 'hour_shift_4','day_shift_4', 'month_shift_4', 'cog_4', 'heading_4']].to_csv('x_test_3.csv', sep=',', index=False)
ais_test[['lat_shift_5', 'lon_shift_5', 'minute_shift_5', 'hour_shift_5','day_shift_5', 'month_shift_5', 'cog_5', 'heading_5', 'cog', 'heading', 'time_diff_1', 'time_diff_2', 'time_diff_3', 'time_diff_4', 'time_diff_5']].to_csv('x_test_4.csv', sep=',', index=False)

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
predictions_df.to_csv('ais_test_predictions_9.csv', index=True)
#pd.DataFrame.to_excel(ais_test.head(1000), "testdata7.xlsx")
#pd.DataFrame.to_excel(ais_train.head(1000), "testdata8.xlsx")
print(datetime.datetime.today().strftime("%HH:%MM %dd"))