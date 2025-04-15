import requests

# Replace these with your ThingSpeak Channel ID and Write API Key
CHANNEL_ID = '2906053'
WRITE_API_KEY = '18NTY0VN3YPMM9FI'

def send_smart_home_data(occupancy, temp, humidity, light):
    # Create the URL for sending data to all fields
    url = f'https://api.thingspeak.com/update?api_key={WRITE_API_KEY}'
    url += f'&field1={occupancy}'
    url += f'&field6={temp}'
    url += f'&field7={humidity}'
    url += f'&field8={light}'
    
    # Send the data using a GET request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        print(f"Data sent successfully! Entry ID: {response.text}")
    else:
        print(f"Failed to send data: {response.status_code}")

# Example: Send sample data to ThingSpeak
# You can modify these values directly in the code as needed
occupancy = 1      # 1 for occupied, 0 for not occupied
temp = 25        # Temperature value
humidity = 23    # Humidity value
light = 5760        # Light intensity value

# Send the data to ThingSpeak
send_smart_home_data(occupancy, temp, humidity, light)