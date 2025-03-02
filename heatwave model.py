from sarima import SARIMA
temperature = [13.64,13.14,12.41,12.13,11.49,11.36,12.29,15,17.81,19.87,21.68,23.1,24.44,25.55,26.75,26.52,27.67,27.59,25.97,25.21,23.35,20.57,18.51,17.39,16.43,15.58,15,14.32,13.77,13.65,14.46,17.34,19.86,21.89,23.67,25.04,25.56,26.79,28.29,29.06,30.17,30.41,30.43,28.96,26.77,24.79,23.15,21.8,20.4,18.86,17.2,16.33,15.48,14.9,15.8,18.67,21.08,23.43,25.3,26.59,27.08,27.98,28.8,29.43,29.85,30.22,29.77,29.05,26.95,25.02,23.05,21.24,19.6,18.99,19.02,18.08,17.43,17.06,17.8,19.28,20.61,22.71,24.03,25.44,26.3,27.37,27.98,28.83,28.96,30.02,29.74,29,25.97,22.6,21.21,19.92,19.03,18.4,17.79,17.23,16.49,16.2,16.79,18.72,20.85,22.54,24.16,25.74,27.26,27.31,28.58,29.16,28.05,27.64,27.79,26.62,24.22,22.23,20.45,20.41,20.17,19.05,17.59,16.8,16.14,17.3,17.88,17.82,18.8,19.54,21.03,23.12,25.15,26.46,27.26,27.36,28.37,27.73,26.48,25.92,23.36,21.04,19.34,17.77,16.71,15.95,15.49,15.02,14.61,14.44,15.03,17.01,19.56,22.13,24.42,26.16,27.36,28.72,28.76,29.77,29.09,28.7,26.85,23.39,22.98,21.97,20,18.73]
humidity = [64.13,68.03,69.08,70.37,71,70.8,70.32,64.83,61.12,61.45,56.98,51.15,46.48,43.92,43.5,43.83,42.85,42.67,43.67,45.17,47.32,51.48,55.05,57.28,59.67,61.77,63.87,67.27,70.65,70.3,68.43,65.97,64.43,59.82,52.73,48.22,45.2,42.62,40.2,37.4,35.93,33.1,33.58,34.3,39.62,51.85,57.68,57.4,58.07,60.9,64.05,66.47,66.48,63.32,66.67,61.67,57.87,50.45,42.05,39.65,39.82,36.85,35.47,35.25,35.97,35,35.02,37.52,42.6,48.07,52.47,57.37,59.83,60.32,61.02,70.02,73.13,71.82,71.93,67.4,64.25,56.38,53.83,48.23,45.97,42.53,40.83,38.5,37.15,32.73,30.82,34.43,45.17,55.42,59.25,61.37,63.42,66.68,69.2,70.77,72.23,73.53,72.58,68.85,60.97,59.45,57.2,52.65,46.35,45.63,38.65,39.23,43.63,49.38,48.77,46.87,48.35,52.28,55.4,51.77,52.02,55.67,58.4,61.8,67.38,72.9,78.63,80.8,78.3,75.1,69.3,60.68,54.35,50.87,45.7,43.1,42.23,42.45,45.33,48.25,51.72,56.28,58.02,62.37,65.42,68.27,70.42,72.15,71.6,73.07,72.67,68.52,62.82,56.95,50.37,42.02,36.92,33.58,31.77,29.95,29.97,32.42,41.18,56.15,61.07,59.3,61.65,62.62]
windspeed = [13.64,13.14,12.41,12.13,11.49,11.36,12.29,15,17.81,19.87,21.68,23.1,24.44,25.55,26.75,26.52,27.67,27.59,25.97,25.21,23.35,20.57,18.51,17.39,16.43,15.58,15,14.32,13.77,13.65,14.46,17.34,19.86,21.89,23.67,25.04,25.56,26.79,28.29,29.06,30.17,30.41,30.43,28.96,26.77,24.79,23.15,21.8,20.4,18.86,17.2,16.33,15.48,14.9,15.8,18.67,21.08,23.43,25.3,26.59,27.08,27.98,28.8,29.43,29.85,30.22,29.77,29.05,26.95,25.02,23.05,21.24,19.6,18.99,19.02,18.08,17.43,17.06,17.8,19.28,20.61,22.71,24.03,25.44,26.3,27.37,27.98,28.83,28.96,30.02,29.74,29,25.97,22.6,21.21,19.92,19.03,18.4,17.79,17.23,16.49,16.2,16.79,18.72,20.85,22.54,24.16,25.74,27.26,27.31,28.58,29.16,28.05,27.64,27.79,26.62,24.22,22.23,20.45,20.41,20.17,19.05,17.59,16.8,16.14,17.3,17.88,17.82,18.8,19.54,21.03,23.12,25.15,26.46,27.26,27.36,28.37,27.73,26.48,25.92,23.36,21.04,19.34,17.77,16.71,15.95,15.49,15.02,14.61,14.44,15.03,17.01,19.56,22.13,24.42,26.16,27.36,28.72,28.76,29.77,29.09,28.7,26.85,23.39,22.98,21.97,20,18.73]
sarimaModel = SARIMA()
sarimaModel.train(temperature, 2, 0, 1, 24)
temperature_predictions = sarimaModel.predict(temperature, 24)
sarimaModel.train(humidity, 2, 0, 1, 24)
humidity_predictions = sarimaModel.predict(humidity, 24)
sarimaModel.train(windspeed, 2, 0, 1, 24)
windspeed_predictions = sarimaModel.predict(windspeed, 24)
import matplotlib.pyplot as plt

# Plot temperature predictions
plt.figure(figsize=(14, 7))
plt.plot(range(len(temperature)), temperature, label='Actual Temperature')
plt.plot(range(len(temperature), len(temperature) + len(temperature_predictions)), temperature_predictions, label='Predicted Temperature', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Predictions')
plt.legend()
#plt.show()

# Plot humidity predictions
plt.figure(figsize=(14, 7))
plt.plot(range(len(humidity)), humidity, label='Actual Humidity')
plt.plot(range(len(humidity), len(humidity) + len(humidity_predictions)), humidity_predictions, label='Predicted Humidity', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Humidity')
plt.title('Humidity Predictions')
plt.legend()
#plt.show()

# Plot wind speed predictions
plt.figure(figsize=(14, 7))
plt.plot(range(len(windspeed)), windspeed, label='Actual Wind Speed')
plt.plot(range(len(windspeed), len(windspeed) + len(windspeed_predictions)), windspeed_predictions, label='Predicted Wind Speed', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Wind Speed')
plt.title('Wind Speed Predictions')
plt.legend()
#plt.show()

from linear_regression import LinearRegression
from matrix import Matrix
inputDataFile = open("C:/Users/tahir/Desktop/M3/Loughborough Uni Test Housing Temps/Weather_hourly.csv", "r")
inputData = [[float(y) for y in x.split(",")[5:7]] for x in inputDataFile.read().split("\n")[1:-1]]
inputDataFile.close()
outputDataFile = open("C:/Users/tahir/Desktop/M3/Loughborough Uni Test Housing Temps/East_AT_hourly.csv", "r")
outputData = [sum([float(y) for y in x.split(",")[1:] if y != "NA"])/(x.count(",") + 1) for x in outputDataFile.read().split("\n")[1:-1]]
outputDataFile.close()
linModel = LinearRegression()
linModel.train(Matrix(inputData), Matrix(outputData).T())
predictions = linModel.predict(Matrix([[temperature_predictions[i], humidity_predictions[i]] for i in range(len(temperature_predictions))]))
print(list(predictions.data))
# Plot linear regression predictions
plt.figure(figsize=(14, 7))
#plt.plot(range(len(temperature[-24:])), temperature[-24:], label='Outdoor Temperature')
#plt.plot(range(len(predictions.data)), predictions.data, label='Indoor Temperature Predictions')
#plt.xlabel('Time')
#plt.ylabel('Temperature')
plt.plot(temperature[-24:], predictions.data)
plt.xlabel('Outdoor Temperature')
plt.ylabel('Indoor Temperature')
plt.title('Indoor Temperature Predictions')
#plt.legend()
plt.show()