import requests

API_KEY = "6d2888301fc71c38471e5735404a12c5"
#CITY = "Bhubaneswar" # e.g., "Bhubaneswar"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url).json()
        
        temp = response['main']['temp']
        humidity = response['main']['humidity']
        
        # Check if rain data exists in the last 1h or 3h
        # OpenWeather returns a 'rain' dict only if it is currently raining
        rain = response.get('rain', {}).get('1h', 0) 
        
        return temp, humidity, rain
        
    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Return safe defaults so the AI doesn't crash
        return 25.0, 50.0, 0.0 