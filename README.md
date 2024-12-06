# Slurred Speech Detection: Multimodal Approach

This project aims to develop a robust system for detecting slurred speech by leveraging both audio features and speech transcriptions. Below is a summary of completed tasks and future directions.

---

## **Completed Work**

1. **Audio Feature Extraction**:
   - Extracted **MFCC features** from audio recordings in TORGO dataset (For Dysarthria), handling inconsistencies like short files.
   - Prepared a structured dataset with features and labels for training.

2. **Audio-Based Classification**:
   - Trained various Bagging and Boosting models on extracted features.
   - Selected the best model by evaluation using metrics such as accuracy, precision, and F1-score.
   - Analyzed feature importance to identify key contributors to the classification.

3. **Deployment**:
   - Deployed the trained model on **Google Cloud Platform** using **Flask API** and **Docker**.
   - Enabled seamless interaction with the model for real-time predictions.

---

## **Planned Work**

1. Explore advanced methods for extracting more detailed insights from both audio and transcribed text.
2. Leverage modern models to integrate audio and text information for improved accuracy.
3. Develop a unified approach combining multiple modalities for enhanced detection capability.

---

## **Next Steps**
The project will expand into incorporating advanced techniques and state-of-the-art models to build a more comprehensive and accurate system for dysarthria detection.
