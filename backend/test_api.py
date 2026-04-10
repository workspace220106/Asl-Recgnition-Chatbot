import requests

url = "http://127.0.0.1:5000/predict"

files = {"image": open(r"C:\Users\Dell\Desktop\python\Test_Images\test_hand.jpg", "rb")}

response = requests.post(url, files=files)

print(response.json())
