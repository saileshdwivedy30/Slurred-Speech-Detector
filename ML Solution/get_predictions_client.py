import os
import requests
import pandas as pd

# Directory containing the audio files
audio_dir = "/Users/saileshdwivedy/PycharmProjects/Slurred Speech Detector/ML Solution/test audio/subset"
# Cloud Run service URL
service_url = "https://lightgbm-audio-predictor-208135024206.us-central1.run.app/predict"

# Function to determine the actual label based on the file name
def determine_actual(file_name):
    if "_FC" in file_name or "_MC" in file_name:
        return "Not Dysarthric"
    else:
        return "Dysarthric"

# List to store results
results = []

# Iterate over all files in the directory
for file_name in os.listdir(audio_dir):
    file_path = os.path.join(audio_dir, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".wav"):
        try:
            print(f"Processing: {file_name}")
            with open(file_path, "rb") as f:
                response = requests.post(service_url, files={"file": f})
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "File": file_name,
                    "Actual": determine_actual(file_name),
                    "Prediction": data.get("prediction", "N/A"),
                    #"Confidence": round(data.get("confidence", 0.0000), 4)  # Round to 4 decimals
                })
            else:
                print(f"Failed to get a prediction for {file_name}: {response.status_code}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert the results to a DataFrame
df_results = pd.DataFrame(results)

# Display the predictions
if not df_results.empty:
#    print("\nPredictions:")
#    print(df_results)

    # Save results to a CSV file for reference
    output_file = "/Users/saileshdwivedy/PycharmProjects/Slurred Speech Detector/ML Solution/predictions"
    df_results = df_results[["File", "Actual", "Prediction"]]
    df_results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
else:
    print("No predictions were made.")
