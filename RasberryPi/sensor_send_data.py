# Simple demo of reading each analog input from the ADS1x15 and printing it to
# the screen.
# Author: Tony DiCola
# License: Public Domain
import time
import requests
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import Adafruit_DHT
# Import the ADS1x15 module.
import Adafruit_ADS1x15

lcd = CharLCD('PCF8574',0x27)
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

write_api_key = 'W9YSPLL5UPWOWMB2'
channel_id = '2913715'
# Create an ADS1115 ADC (16-bit) instance.
#adc = Adafruit_ADS1x15.ADS1115()
#CH1 = Variable resistor
#CH2 = Sound 
#CH3 = Light

# Or create an ADS1015 ADC (12-bit) instance.
adc = Adafruit_ADS1x15.ADS1115()

# Note you can change the I2C address from its default (0x48), and/or the I2C
# bus by passing in these optional parameters:
#adc = Adafruit_ADS1x15.ADS1015(address=0x49, busnum=1)

# Choose a gain of 1 for reading voltages from 0 to 4.09V.
# Or pick a different gain to change the range of voltages that are read:
#  - 2/3 = +/-6.144V
#  -   1 = +/-4.096V
#  -   2 = +/-2.048V
#  -   4 = +/-1.024V
#  -   8 = +/-0.512V
#  -  16 = +/-0.256V
# See table 3 in the ADS1015/ADS1115 datasheet for more info on gain.
GAIN = 1
#setup GPIO in, out, DHTSensor for temp and humidtity


#temp and humid storing
# try:
#     Adafruit_DHT.DH11=1
# except NotImplmentedError:
#     Adafruit_DHT.DHT11 = None
        
#count = 0

print('Reading ADS1x15 values, press Ctrl-C to quit...')
# Print nice channel column headers.
print('| {0:>6} | {1:>6} | {2:>6} | {3:>6} |'.format(*range(4)))
print('|Adafruit|Resistor|  Sound |  Light |humidity|temperature|')
print('-' * 58)
# Main loop.
try:
    while True:
        # Read all the ADC channel values in a list.
        values = [0]*4
        for i in range(4):
            # Read the specified ADC channel using the previously set gain value.
            values[i] = adc.read_adc(i, gain=GAIN)
            
            # Note you can also pass in an optional data_rate parameter that controls
            # the ADC conversion time (in samples/second). Each chip has a different
            # set of allowed data rate values, see datasheet Table 9 config register
            # DR bit values.
            #values[i] = adc.read_adc(i, gain=GAIN, data_rate=128)
            # Each value will be a 12 or 16 bit signed integer value depending on the
            # ADC (ADS1015 = 12-bit, ADS1115 = 16-bit).
        humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)              
        
        
        if values[1] > 0:
            occupancy = 1
        else:
            occupancy = 0
        print('| {0:>6} | {1:>6} | {2:>6} | {3:>6} |'.format(*values) + f"  {humidity:0.1f}  |    {temperature:0.1f}   |")
        
        requests.post(f'https://api.thingspeak.com/update?api_key={write_api_key}&field6={temperature}&field7={humidity}&field8={values[3]}&field1={occupancy}')
        # Display temperature and humidity on the LCD
        lcd.clear()
        lcd.cursor_pos = (0,0)
        lcd.write_string(f'Temp: {temperature:.1f}C')
        lcd.cursor_pos=(1,0) #Move to the second line
        lcd.write_string(f'Hum: {humidity:.1f}%')
        time.sleep(4)
        lcd.clear()
        lcd.cursor_pos = (0,0)
        #lcd.write_string(f'Sound: {values[2]}')
        lcd.cursor_pos=(1,0) #Move to the second line
        lcd.write_string(f'Light: {values[3]}')
        time.sleep(4)
        
        lcd.clear()
        lcd.write_string(f'Resistor: {values[1]}')
        lcd.cursor_pos=(1,0)
        lcd.write_string(f'Occupancy: {occupancy}')
        
        # Pause for 15s in total.
        time.sleep(15)
except KeyboardInterrupt:
    lcd.clear()
    lcd.cursor_pos = (0,3)
    lcd.write_string('THE END')
    time.sleep(2)
    lcd.clear()