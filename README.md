# Machine-Learning Keyword Detection on Arduino Nano 33 BLE Sense

This project develops a TinyML keyword spotter on the Arduino Nano 33 BLE Sense using custom audio data, augmentation, and spectrogram-based features (Microfrontend/STFT). A lightweight CNN is trained and quantized for real-time, on-device inference, evaluating tradeoffs between accuracy, model size, and latency.

---

## Authors
- **Christian Bammann** – Contributor of data collection/preprocessing, CNN model development, and co-author of the written final project report.  
- **Ryan Monroe** – Contributor of data collection/preprocessing, embedded system deployment, and co-author of the written final project report.

---

## Contents

| Folder                                                                                     | Description                                                 |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `embedded`                                 | All C/C++ code used to deploy model onto the embedded platform                      |
| `final_model`                                    | Python notebook with ANN Model                              |
| `training`                                          | Final trained models in h5, quantized tflite, and cc file formats                               |
| `README.md`                                                                              | Project Summary                                             |
  
---

## Results

| Model                               | Mean Squared Error (MSE)    | Mean Absolute Error (MAE)   | R² Score                    | 
|-------------------------------------|-----------------------------|-----------------------------|-----------------------------|
| Linear Regression (LR)              | 0.176                       | 0.263                       | 0.790                       |
| Support Vector Machine (SVM)        | 0.107                       | 0.156                       | 0.872                       |
| Artificial Neural Network (ANN)     | 0.125                       | 0.194                       | 0.861                       |

---

## Analysis
Placeholder

## Key Insights
Placeholder

---

## Conclusion
Placeholder
