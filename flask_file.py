from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

# Load the dataset
data_set = pd.read_csv('C:/Users/User/Downloads/drone_sys-20230531T103434Z-001/drone_sys/final_data.csv')

# Separate numeric and string columns
numeric_columns = ['flight radius(0m-10000m)', 'flight height(0m-10000m)', 'operations per head(0-100)', 'flight time(0min-100min)', 'payload capacity(0kg-20kg)', 'range(0km-35km)', 'camera quality(0-50megapixels)', 'wind resistance(0-100km/h)', 'noise level(0db-100db)', 'budget(0$-20000$)', 'Data storage(0gb-512gb)', 'durability(0-10rating)', 'weather resistance(0-10rating)', 'speed(0-150km/h)', 'GPSaccuracy(0m-30m)']
string_columns = ['obstacle avoidance(yes=\'1\'/no=\'0\')', 'regulatory compilance(yes=\'1\'/no=\'0\')', 'integration with systems(yes=\'1\'/no=\'0\')']

# Convert string columns to binary features using one-hot encoding
string_features = pd.get_dummies(data_set[string_columns])

# Combine numeric and string features
features = pd.concat([data_set[numeric_columns], string_features], axis=1)

# Split the dataset into features and target variable
target = data_set['Drone Model']

# Feature scaling for numeric features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

app = Flask(__name__, template_folder="C:/Users/User/Downloads/drone_sys-20230531T103434Z-001/drone_sys/template1")
client = MongoClient('mongodb://localhost:27017/')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', features=features)

@app.route('/result', methods=['GET', 'POST'])
def result():
    user_requirements =  {}
    for criteria in features.columns:
        value = request.form.get(criteria, 0)
        user_requirements[criteria] = float(value)

    # Scale user requirements
    user_requirements_scaled = scaler.transform(pd.DataFrame([user_requirements]))

    # Calculate the Euclidean Distance between user requirements and drone features for each drone model
    drone_scores = {}
    for idx, drone_features in enumerate(scaled_features):
        dist = distance.euclidean(user_requirements_scaled[0], drone_features)
        drone_scores[idx] = dist

    # Select the drone models with the minimum distance as the best choices
    best_drone_models = []
    best_scores = []
    sorted_drone_scores = sorted(drone_scores.items(), key=lambda x: x[1])
    for idx, score in sorted_drone_scores[:3]:
        best_drone_models.append(data_set['Drone Model'][idx])
        best_scores.append(score)

    # Store the results in the MongoDB database
    db = client['drone_management_system']
    collection = db['drone_collection_system']
    result_data = {
        'best_drone': best_drone_models[0],
        'score': best_scores[0],
        'second_best_drone': best_drone_models[1],
        'second_score': best_scores[1],
        'third_best_drone': best_drone_models[2],
        'third_score': best_scores[2]
    }
    collection.insert_one(result_data)

    return render_template('result.html',
                           best_drone=best_drone_models[0],
                           score=best_scores[0],
                           second_best_drone=best_drone_models[1],
                           second_score=best_scores[1],
                           third_best_drone=best_drone_models[2],
                           third_score=best_scores[2])
if __name__ == '__main__':
    app.run(debug=True)