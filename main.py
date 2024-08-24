import model
import pandas as pd 
import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Weather Prediction")
root.geometry("400x400")

image_path = "Weather.png"  
image = Image.open(image_path)
image = image.resize((200, 100)) 
weather_image = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=weather_image)
image_label.pack(pady=10)

heading_label = tk.Label(root, text="Weather Predictor", font=("Arial", 18))
heading_label.pack()

precipitation_label = tk.Label(root, text="Enter Precipitation (%):")
precipitation_label.pack()

entry_precipitation = tk.Entry(root)
entry_precipitation.pack()

temp_label = tk.Label(root, text="Enter Temperature (°C):")
temp_label.pack()

entry_temp = tk.Entry(root)
entry_temp.pack()

wind_label = tk.Label(root, text="Enter Wind Speed(km/h):")
wind_label.pack()

entry_wind = tk.Entry(root)
entry_wind.pack()


def predict_weather():
    precipitation = entry_precipitation.get()
    temperature = entry_temp.get()
    wind = entry_wind.get()

    print(f"precipitation: {precipitation} %")
    print(f"Temperature: {temperature} °C")
    print(f"Wind: {wind} km/h")

    x = float(precipitation)
    y = float(temperature)
    z = float(wind)

    predictor = model.predict('random_forest_model.pkl')
    pred = predictor.value([[x, y, z]])
    # realtime = int(pred)
    realtime = pred
    print(realtime)

    if(realtime <= 0.8):
        weather_prediction = "Drizzling"
    elif(realtime > 0.8 and realtime <= 1.2):
        weather_prediction = "Fog"
    elif(realtime > 1.2 and realtime <= 2.4):
        weather_prediction = "Rainy"
    elif(realtime > 2.4 and realtime <= 3.5):
        weather_prediction = "Sunny"
    elif(realtime > 3.5):
        weather_prediction = "Snow"
    else:
        weather_prediction = "Invalid input!"

    print("Predicted weather: ", weather_prediction)
    output_label.config(text=f"Weather Prediction: {weather_prediction}")

predict_button = tk.Button(root, text="Predict Weather", command=predict_weather)
predict_button.pack(pady=10)

output_label = tk.Label(root, text="Predicted Weather: ", font=("Arial", 12))
output_label.pack()

root.mainloop()

