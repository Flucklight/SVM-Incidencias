import requests

URL = "http://127.0.0.1:8000/clasificar"

payload = {
    "comentario": "Servicio de Atencion Una disculpa me comunico por el presente para informar sobre el cable HDMI dañado"
}

response = requests.post(URL, json=payload, timeout=30)

print("Status:", response.status_code)
print("Respuesta:")
print(response.json())