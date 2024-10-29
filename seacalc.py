import pandas as pd
import math
import numpy as np
import datetime
from io import StringIO
import random
from geopy.distance import geodesic
import searoute
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
schedules = schedules.sort_values(by=['datetime','vesselId'])

vessels = pd.read_csv('vessels.csv', sep='|')

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

ais_test = pd.merge_asof(ais_test, schedules, by='vesselId', on='datetime', direction='forward', suffixes=('', '_sch'))

features = ['vesselId', 'hour', 'day', 'month', 'minute', 'vesselType', 'yearBuilt', 'length', 'breadth', 'CEU', 'DWT', 'GT', 'portLatitude', 'portLongitude', 'hour_sch', 'day_sch', 'month_sch', 'minute_sch'] + liste#'vesselType', 'yearBuilt', , 'portLatitude', 'portLongitude', 'sch_hour', 'sch_minute']#['cog', 'sog', 'rot', 'heading', 'navstat', 'latitude', 'longitude', 'vesselId', 'portId', 'hour', 'day', 'month', 'year', 'minute']
print("features:", features)
target = ['longitude', 'latitude']  # Predicting next position (latitude, longitude)


        
def feature_engineering(data):
    data['vesselType'].fillna(83, inplace=True) 
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
ais_test = feature_engineering(ais_test)[features]

test_predictions = pd.read_csv("ais_test_predictions_7.csv", sep=",")
test_predictions['longitude'] = test_predictions['longitude_predicted']
test_predictions['latitude'] = test_predictions['latitude_predicted']
predictions_df = test_predictions[['longitude', 'latitude']]
last = ais_test.join(predictions_df)
X_test2 = ais_test.join(predictions_df)

for i in range(1, timesteps):  # Only 3 previous time steps
    X_test2[f'lat_shift_{i}'] = X_test2.groupby('vesselId')['latitude'].shift(i)
    X_test2[f'lon_shift_{i}'] = X_test2.groupby('vesselId')['longitude'].shift(i)

    X_test2[f'lat_shift_{i}'] = np.where(X_test2[f'lat_shift_{i}'].isna(), last[f'lat_shift_{i}'], X_test2[f'lat_shift_{i}'])
    X_test2[f'lon_shift_{i}'] = np.where(X_test2[f'lon_shift_{i}'].isna(), last[f'lon_shift_{i}'], X_test2[f'lon_shift_{i}'])

# Helper functions for distance, bearing, and destination point calculations
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in km

def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360  # Normalize to 0-360 degrees

def destination_point(lat, lon, distance_km, bearing_deg):
    R = 6371  # Radius of Earth in kilometers
    bearing = math.radians(bearing_deg)
    lat, lon = math.radians(lat), math.radians(lon)

    new_lat = math.asin(math.sin(lat) * math.cos(distance_km / R) +
                        math.cos(lat) * math.sin(distance_km / R) * math.cos(bearing))
    new_lon = lon + math.atan2(math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat),
                               math.cos(distance_km / R) - math.sin(lat) * math.sin(new_lat))

    return math.degrees(new_lat), math.degrees(new_lon)

# Sample DataFrame with necessary columns
# Calculate speed and bearing for each vessel entry
df = pd.DataFrame()
df['longitude_predicted'] = None
df['latitude_predicted'] = None

all_predictions = {}
for idx, row in X_test2.iterrows():
    vessel_id = row['vesselId']
    
    # Retrieve the last known predicted coordinates, if available
    if True:#not vessel_id in all_predictions:
        lat1, lon1 = row['lat_shift_1'], row['lon_shift_1']
        lat2, lon2 = row['lat_shift_2'], row['lon_shift_2']
    elif(len(all_predictions[vessel_id])==1):
        lat1, lon1 = all_predictions[vessel_id][0][0], all_predictions[vessel_id][0][1]
        lat2, lon2 = row['lat_shift_2'], row['lon_shift_2']
    else:
        lat1, lon1 = all_predictions[vessel_id][len(all_predictions[vessel_id])-1][0], all_predictions[vessel_id][len(all_predictions[vessel_id])-1][1]
        lat2, lon2 = all_predictions[vessel_id][len(all_predictions[vessel_id])-2][0], all_predictions[vessel_id][len(all_predictions[vessel_id])-2][1]
    port_lat, port_lon = row['portLatitude'], row['portLongitude']
    
    # Step 1: Use searoute to calculate the distance between the last two known positions
    distance_km = haversine(lat2, lon2, lat1, lon1)  # Distance in kilometers
    timediff_2_hours = row['time_diff_2'] / 3600  # Convert seconds to hours
    speed_kmh = distance_km / timediff_2_hours  # Speed in km/h

    bearing_to_port = calculate_bearing(lat1, lon1, port_lat, port_lon)
    timediff_1_hours = row['time_diff_1'] / 3600
    distance_to_travel = speed_kmh * timediff_1_hours
    
    origin = [lon1, lat1]
    destination = [port_lon, port_lat]
    print(origin, destination)
    route = searoute.searoute(origin, destination)  # This returns a GeoJSON LineString
    
    # Extract the coordinates from the GeoJSON route
    route_coords = route['geometry']['coordinates']
    
    # Step 4: Traverse the route coordinates to find the predicted position
    cumulative_distance = 0
    predicted_lat, predicted_lon = lat1, lon1  # Initialize with the starting point

    for i in range(1, len(route_coords)):
        # Calculate the segment distance
        point_a = (route_coords[i-1][1], route_coords[i-1][0])  # (lat, lon)
        point_b = (route_coords[i][1], route_coords[i][0])  # (lat, lon)
        segment_distance = geodesic(point_a, point_b).kilometers

        # Add segment distance to cumulative distance
        cumulative_distance += segment_distance

        # Check if cumulative distance has reached or exceeded the distance to travel
        if cumulative_distance >= distance_to_travel:
            predicted_lat, predicted_lon = point_b
            break
        else:
            predicted_lat, predicted_lon = point_b  # Update to last point reached

    # Update the DataFrame with the predicted coordinates
    df.at[idx, 'longitude_predicted'] = predicted_lon
    df.at[idx, 'latitude_predicted'] = predicted_lat
    if vessel_id not in all_predictions:
        all_predictions[vessel_id] = []
    all_predictions[vessel_id].append([predicted_lat, predicted_lon])

    
df.to_csv('ais_test_predictions_11.csv', index=True)
    #print(f"Predicted next coordinates for vessel {row['vesselId']}: Latitude: {predicted_lat}, Longitude: {predicted_lon}")