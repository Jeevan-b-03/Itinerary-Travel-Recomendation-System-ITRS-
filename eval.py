import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# API URLs and Keys
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
HOTELS_BASE_URL = 'https://api.liteapi.travel/v3.0/data/hotels'
GEOCODING_URL = 'https://geocode.maps.co/search'
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'

GEMINI_API_KEY = 'AIzaSyCcgDpa8pk3VkymAcayCnLlej8QH_ylkEs'
HOTEL_API_KEY = 'sand_6e380dc3-4178-4799-bbc2-81964b633767'
GEOCODING_API_KEY = '66e6e69de7b65672694704ftkef8443'
WEATHER_API_KEY = '8d106c8a0de24d19a70161452241809'

# Load datasets
df = pd.read_csv('tourism_dataset.csv')
df1 = pd.read_csv('worldcities.csv')

# KMeans clustering based on latitude and longitude
kmeans = KMeans(n_clusters=5, random_state=0)
df1['loc_clusters'] = kmeans.fit_predict(df1[['lat', 'lng']])

# Calculate Silhouette Score for KMeans clustering
sil_score = silhouette_score(df1[['lat', 'lng']], df1['loc_clusters'])
print(f'Silhouette Score for KMeans clustering: {sil_score:.4f}')

# Recommend similar cities based on the cluster
def recommend_similar_cities(input_city, df1):
    input_city_lower = input_city.lower()
    df1['city_lower'] = df1['city'].str.lower()  # Create a new column for lowercase city names

    cluster = df1.loc[df1['city_lower'] == input_city_lower, 'loc_clusters']
    
    if not cluster.empty:
        cluster = cluster.iloc[0]
        cities_in_cluster = df1.loc[df1['loc_clusters'] == cluster, 'city']
        recommended_cities = [city for city in cities_in_cluster if city.lower() != input_city_lower]
        return recommended_cities[:5] if recommended_cities else ["No other cities found in the same cluster."]
    else:
        return ["City not found in the dataset."]


# Fetch location coordinates using Geocoding API
def get_location_coordinates(location):
    params = {'q': location}
    response = requests.get(GEOCODING_URL, params=params)
    
    if response.status_code == 200:
        location_data = response.json()
        if location_data:
            latitude = location_data[0]['lat']
            longitude = location_data[0]['lon']
            return latitude, longitude
        else:
            print(f"No coordinates found for the location: {location}")
            return None, None
    else:
        print(f"Failed to retrieve coordinates for {location}. Status code: {response.status_code}")
        return None, None

# Fetch hotels using the LiteAPI
def get_hotels_and_similarity(preferred_hotel_name, location):
    latitude, longitude = get_location_coordinates(location)
    if latitude and longitude:
        params = {'latitude': latitude, 'longitude': longitude, 'radius': 1000}
        headers = {'X-API-Key': HOTEL_API_KEY, 'accept': 'application/json'}

        response = requests.get(HOTELS_BASE_URL, headers=headers, params=params)
        if response.status_code == 200:
            hotels = response.json().get('data', [])
            hotel_names = [hotel['name'] for hotel in hotels]
            all_hotel_names = [preferred_hotel_name] + hotel_names

            # TF-IDF and Cosine Similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_hotel_names)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Sort and select top 5 hotels based on similarity
            similarity_scores = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
            top_hotels = similarity_scores[:5]

            print(f"Top 5 hotels similar to '{preferred_hotel_name}' near {location}:")
            for idx, score in top_hotels:
                hotel = hotels[idx]
                print(f"Hotel Name: {hotel['name']}, Similarity Score: {score:.2f}")
        else:
            print(f"Failed to fetch hotels data. Status code: {response.status_code}")
    else:
        print("Could not determine the location's coordinates.")

# Test City Recommendation and display similar cities
input_city = 'chennai'
recommended_cities = recommend_similar_cities(input_city, df1)
print(f"Cities similar to {input_city}: {recommended_cities}")

# Test hotel recommendations
preferred_hotel_name = 'the'
location = 'chennai'
get_hotels_and_similarity(preferred_hotel_name, location)

import matplotlib.pyplot as plt

# Scatter plot of the clusters
plt.figure(figsize=(10, 6))

# Assign colors to each cluster
colors = ['red', 'blue', 'green', 'purple', 'orange']

for cluster in range(kmeans.n_clusters):
    cluster_data = df1[df1['loc_clusters'] == cluster]
    plt.scatter(cluster_data['lng'], cluster_data['lat'], 
                label=f'Cluster {cluster}', 
                color=colors[cluster % len(colors)], 
                alpha=0.6)

# Highlight the input city, if it exists in the dataset
if input_city.lower() in df1['city_lower'].values:
    input_city_data = df1[df1['city_lower'] == input_city.lower()]
    plt.scatter(input_city_data['lng'], input_city_data['lat'], 
                color='black', 
                label=f'Input City: {input_city.capitalize()}', 
                edgecolor='yellow', 
                s=100)

# Add labels and legend
plt.title('Clusters of Cities Based on Latitude and Longitude', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(loc='best')
plt.grid(alpha=0.4)

# Display the plot
plt.show()
