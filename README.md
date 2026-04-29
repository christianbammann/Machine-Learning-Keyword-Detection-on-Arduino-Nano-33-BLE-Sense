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

| Model                               | Train Acc    | Val Acc   | Test Acc                   | 
|-------------------------------------|-----------------------------|-----------------------------|-----------------------------|
| Model A (Microfrontend-based Spectrogram)              | 73.31%                       | 87.95%                       | 89.41%                       |
| Model B (STFT-based Spectrogram)        | 98.13%                       | 95.52%                       | 94.52%                       |

