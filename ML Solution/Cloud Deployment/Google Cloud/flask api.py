from flask import Flask, request, jsonify
import lightgbm as lgb
import librosa
import numpy as np
import os

# Load the LightGBM model
lgbm_model = lgb.Booster(model_file='lightgbm_model.txt')

# Initialize the Flask app
app = Flask(__name__)

# Feature extraction function
def extract_mfcc(filepath, sr=22050, n_mfcc=13, target_length=2048):
    try:
        # Load audio file
        audio, _ = librosa.load(filepath, sr=sr)

        # Pad audio if too short
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        raise ValueError(f"Error processing file {filepath}: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the file temporarily
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)

        # Extract features
        features = extract_mfcc(filepath).reshape(1, -1)

        # Get prediction
        prediction = (lgbm_model.predict(features) > 0.5).astype(int)[0]
        confidence = lgbm_model.predict(features).max()

        result = {
            'prediction': 'Dysarthric' if prediction == 1 else 'Not Dysarthric',
            'confidence': round(confidence, 2)
        }

        # Clean up temporary file
        os.remove(filepath)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
