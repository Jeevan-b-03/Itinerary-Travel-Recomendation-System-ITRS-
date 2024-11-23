import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# API URLs
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
HOTELS_BASE_URL = 'https://api.liteapi.travel/v3.0/data/hotels'
GEOCODING_URL = 'https://geocode.maps.co/search'
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'

# API Keys
GEMINI_API_KEY = 'AIzaSyCcgDpa8pk3VkymAcayCnLlej8QH_ylkEs'
HOTEL_API_KEY = 'sand_6e380dc3-4178-4799-bbc2-81964b633767'
GEOCODING_API_KEY = '66e6e69de7b65672694704ftkef8443'
WEATHER_API_KEY = '8d106c8a0de24d19a70161452241809'

# Set page config first
st.set_page_config(page_title="Itinerary & Hotel Recommender", layout="wide")

# Load dataset (use st.cache_data instead of st.cache)
@st.cache_data
def load_data():
    return pd.read_csv('tourism_dataset.csv')

df = load_data()

# Load cities dataset for clustering
df1 = pd.read_csv('worldcities.csv')

# Perform KMeans clustering based on lat and lng columns
kmeans = KMeans(n_clusters=5, random_state=0)
df1['loc_clusters'] = kmeans.fit_predict(df1[['lat', 'lng']])

# Function to recommend cities based on latitude and longitude clusters
def recommend_similar_cities(input_city, df1):
    input_city_lower = input_city.lower()
    df1['city_lower'] = df1['city'].str.lower()  # Create a new column for lowercase city names
    
    cluster = df1.loc[df1['city_lower'] == input_city_lower, 'loc_clusters']
    
    if not cluster.empty:
        cluster = cluster.iloc[0]
        cities_in_cluster = df1.loc[df1['loc_clusters'] == cluster, 'city']
        recommended_cities = [city for city in cities_in_cluster if city.lower() != input_city_lower]
        
        # Limit recommendations to 5 cities
        return recommended_cities[:5] if recommended_cities else ["No other cities found in the same cluster."]
    else:
        return ["City not found in the dataset."]

# Function to get latitude and longitude of a location using Geocoding API
def get_location_coordinates(location):
    params = {'q': location}  # Query the location name
    response = requests.get(GEOCODING_URL, params=params)
    
    if response.status_code == 200:
        location_data = response.json()
        if location_data and len(location_data) > 0:
            # Extract the first matching location's latitude and longitude
            latitude = location_data[0]['lat']
            longitude = location_data[0]['lon']
            return latitude, longitude
        else:
            st.error(f"No coordinates found for the location: {location}")
            return None, None
    else:
        st.error(f"Failed to retrieve coordinates for {location}. Status code: {response.status_code}")
        return None, None


# Sidebar
st.sidebar.image("VITLOGO.png", width=200)
st.sidebar.title("Itinerary Travel Recommender")
st.sidebar.write("Made by Janani and Jeevan")
st.sidebar.write("Choose one of the options:")

# Toggle between itinerary, similar city recommendation, and hotel recommendation
page_option = st.sidebar.radio("Select Mode", ["Itinerary Recommendation", "Hotel Recommendation"])

if page_option == "Itinerary Recommendation":
    st.title("Itinerary Recommender")
    st.markdown("### Enter your trip details:")
    
    category = st.selectbox("Category:", df['Category'].unique())
    num_people = st.number_input("Number of people:", min_value=1, step=1)
    budget = st.selectbox("Budget:", ['Low', 'Medium', 'High'])
    num_days = st.number_input("Number of days (optional):", min_value=1, step=1, format="%d", value=None)
    destination = st.text_input("Destination (optional):")
    weather_option = st.checkbox("Include Weather in Itinerary", value=False)

    # Function to get weather data
    def get_weather(destination):
        weather_url = f"{WEATHER_API_URL}?key={WEATHER_API_KEY}&q={destination}&aqi=no"
        response = requests.get(weather_url)

        if response.status_code == 200:
            weather_data = response.json()
            location = weather_data['location']['name']
            current_weather = weather_data['current']
            temp_c = current_weather['temp_c']
            condition = current_weather['condition']['text']
            wind_kph = current_weather['wind_kph']
            humidity = current_weather['humidity']

            return {
                "location": location,
                "temperature": temp_c,
                "condition": condition,
                "wind": wind_kph,
                "humidity": humidity
            }
        else:
            return None

    # Function to recommend destination and number of days based on dataset
    def recommend_from_dataset(category, num_people, budget):
        filtered_df = df[(df['Category'] == category) & (df['Accommodation_Available'] == 'Yes')]

        budget_ranges = {'Low': (0, 10000), 'Medium': (10001, 50000), 'High': (50001, float('inf'))}
        min_revenue, max_revenue = budget_ranges.get(budget, (0, float('inf')))

        filtered_df = filtered_df[(filtered_df['Revenue'] >= min_revenue) & (filtered_df['Revenue'] <= max_revenue)]

        if not filtered_df.empty:
            avg_visitor = filtered_df['Visitors'].mean()
            recommended_df = filtered_df[filtered_df['Visitors'] >= avg_visitor]

            if not recommended_df.empty:
                recommended_location = recommended_df.sample(1).iloc[0]
                recommended_destination = recommended_location['Country']
                recommended_num_days = 3
                return recommended_destination, recommended_num_days

        return None, None

    # Function to get detailed itinerary using Gemini API
    def get_detailed_itinerary(destination, num_days, num_people, budget=None, free_text=None):
        if free_text:
            prompt = free_text
        else:
            prompt = f"I am planning a trip to {destination} for {num_days} days with {num_people} people. My budget is {budget}. Please provide a detailed itinerary with times, transport, and costs."

        if weather_option:
            weather_data = get_weather(destination)
            if weather_data:
                weather_info = f" The weather forecast is {weather_data['condition']} with a temperature of {weather_data['temperature']}°C."
                prompt += weather_info

        data = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            itinerary_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return itinerary_text
        else:
            st.error(f"Failed to fetch itinerary. Status code: {response.status_code}")
            return None

    # Buttons for recommendations and itinerary generation
    if st.button("Recommend Destination and Number of Days"):
        recommended_destination, recommended_num_days = recommend_from_dataset(category, num_people, budget)
        if recommended_destination and recommended_num_days:
            st.write(f"Recommended Destination: {recommended_destination}")
            st.write(f"Recommended Number of Days: {recommended_num_days}")
        else:
            st.write("No recommendations available based on the provided criteria.")

    if st.button("Generate Itinerary"):
        if not destination or not num_days:
            recommended_destination, recommended_num_days = recommend_from_dataset(category, num_people, budget)
            if not destination:
                destination = recommended_destination
            if not num_days:
                num_days = recommended_num_days

        if destination and num_days:
            itinerary = get_detailed_itinerary(destination, num_days, num_people, budget=budget)
            if itinerary:
                st.write("## Your Itinerary:")
                st.write(itinerary)
        else:
            st.error("Please provide destination and number of days to generate the itinerary.")

    st.title("Similar City Recommender")
    input_city = st.text_input("Enter a city name:")
    
    if st.button("Recommend Similar Cities"):
        similar_cities = recommend_similar_cities(input_city, df1)
        st.write("### Recommended Similar Cities:")
        for city in similar_cities:
            st.write(city)

elif page_option == "Hotel Recommendation":
    st.title("Hotel Recommender")
    location = st.text_input("Enter location for hotel search:")
    preferred_hotel_name = st.text_input("Enter your preferred hotel name:")

    # Function to fetch hotel recommendations using LiteAPI
    def get_hotels_and_similarity(preferred_hotel_name, location):
        latitude, longitude = get_location_coordinates(location)  # Now defined
        if latitude and longitude:
            params = {'latitude': latitude, 'longitude': longitude, 'radius': 1000}
            headers = {'X-API-Key': HOTEL_API_KEY, 'accept': 'application/json'}

            response = requests.get(HOTELS_BASE_URL, headers=headers, params=params)
            if response.status_code == 200:
                hotels = response.json().get('data', [])
                hotel_names = [hotel['name'] for hotel in hotels]
                all_hotel_names = [preferred_hotel_name] + hotel_names

                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(all_hotel_names)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                similarity_scores = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
                top_hotels = similarity_scores[:5]

                st.write(f"Top hotels near {location}:")
                for idx, score in top_hotels:
                    hotel = hotels[idx]
                    st.write(f"Hotel Name: {hotel['name']}, Similarity: {score:.2f}")
            else:
                st.error("Failed to fetch hotel data.")
        else:
            st.error("Could not determine the location's coordinates.")

    if st.button("Get Hotel Recommendations"):
        get_hotels_and_similarity(preferred_hotel_name, location)

# Background image
#background_image = "https://png.pngtree.com/background/20231030/original/pngtree-opulent-lounge-in-a-five-star-hotel-grand-wooden-furnishings-enhance-picture-image_5792865.jpg"
background_image = "https://png.pngtree.com/background/20230403/original/pngtree-city-%E2%80%8B%E2%80%8Bnight-silence-picture-image_2273664.jpg"
#background_image = "https://images.pexels.com/photos/450441/pexels-photo-450441.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

# Adding background image styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({background_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        height: 100vh;
    }}
    </style>
    """,
    unsafe_allow_html=True
)