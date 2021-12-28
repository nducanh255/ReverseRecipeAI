from flask import json
from flask.scaffold import F
import requests
import time
import base64
import json

start = time.time()
image = base64.b64encode(open('image.jpg', 'rb').read()).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": image})
resp = requests.post('https://inverse-cooking.herokuapp.com/predict', data=payload, headers=headers)

print(resp.text)
print(time.time() - start)

