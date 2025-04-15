import requests
import RPi.GPIO as GPIO
import time
import json

# --- Configuration ---
THINGSPEAK_CHANNEL_ID = '2906053'  # Replace with your ThingSpeak Channel ID
THINGSPEAK_READ_API_KEY = 'EOXRXL9U5XQQXXEJ' # Replace with your Read API Key (if channel is private, otherwise leave empty '')

# Map ThingSpeak fields to Raspberry Pi GPIO pins (BCM numbering) ---
GPIO_PIN_MAP = {
    'field3': 23,  # Air con_on
    'field4': 24,  # Humidity _on
    'field5': 25   # Lights_on
}

EXTRA_PINS_TO_SET_LOW = [10,8]

CHECK_INTERVAL_SECONDS = 15 # How often to check ThingSpeak (in seconds)

# ThingSpeak API URL for the last feed
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
if THINGSPEAK_READ_API_KEY:
    THINGSPEAK_URL += f"?api_key={THINGSPEAK_READ_API_KEY}"

# --- GPIO Setup ---
def setup_gpio():
    """Sets up GPIO pins."""
    GPIO.setmode(GPIO.BCM) # Use Broadcom pin numbering
    GPIO.setwarnings(False) # Disable warnings
    
    controlled_pins = list(GPIO_PIN_MAP.values())
    all_output_pins = list(set(controlled_pins + EXTRA_PINS_TO_SET_LOW))
    if all_output_pins:
        GPIO.setup(all_output_pins, GPIO.OUT) # Set all mapped pins as output
        GPIO.output(all_output_pins, GPIO.LOW) # Turn all LEDs off initially
        print(f"GPIO pins {all_output_pins} set up as outputs and turned off.")
    else:
        print("Warning: No GPIO pins defined in GPIO_PIN_MAP.")

# --- ThingSpeak Fetch Function ---
def get_latest_thingspeak_data():
    """Fetches the latest data point from ThingSpeak."""
    try:
        response = requests.get(THINGSPEAK_URL, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        print(f"Successfully fetched data: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ThingSpeak: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding ThingSpeak JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None # Return None if fetching failed

# --- LED Control Function ---
def control_leds(data):
    """Controls LEDs based on fetched ThingSpeak data."""
    if not data:
        print("No data received, skipping LED control.")
        return

    for field, pin in GPIO_PIN_MAP.items():
        try:
            # ThingSpeak fields can be None or missing if not updated recently
            field_value_str = data.get(field)

            if field_value_str is not None:
                # Compare the string value
                if field_value_str == '1':
                    GPIO.output(pin, GPIO.HIGH) # Turn LED ON
                    print(f"Field '{field}' is 1. Turning ON GPIO {pin}.")
                elif field_value_str == '0':
                    GPIO.output(pin, GPIO.LOW)  # Turn LED OFF
                    print(f"Field '{field}' is 0. Turning OFF GPIO {pin}.")
                else:
                     # Optional: Turn off if value is unexpected, or maintain state
                     GPIO.output(pin, GPIO.LOW)
                     print(f"Field '{field}' has unexpected value '{field_value_str}'. Turning OFF GPIO {pin}.")

            else:
                 # Optional: Decide what to do if a field is missing (e.g., turn off)
                 GPIO.output(pin, GPIO.LOW)
                 print(f"Field '{field}' not found in latest data. Turning OFF GPIO {pin}.")

        except KeyError:
            print(f"Error: GPIO pin {pin} for field '{field}' not setup correctly or invalid.")
        except Exception as e:
            print(f"An error occurred controlling GPIO {pin} for field '{field}': {e}")

# --- Main Loop ---
if __name__ == "__main__":
    setup_gpio()
    try:
        while True:
            print(f"\nChecking ThingSpeak at {time.ctime()}...")
            latest_data = get_latest_thingspeak_data()
            control_leds(latest_data)
            print(f"Waiting for {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        print("Cleaning up GPIO...")
        GPIO.cleanup() # Reset GPIO pins on exit
        print("GPIO cleanup complete.")
