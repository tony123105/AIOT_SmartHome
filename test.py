import requests

write_api_key = 'W9YSPLL5UPWOWMB2'
channel_id = '2913715'

air_on = 0
hum_on = 1
light_on = 0
requests.post(f'https://api.thingspeak.com/update?api_key={write_api_key}&field3={air_on}&field4={hum_on}&field5={light_on}')
