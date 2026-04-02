# Neem-leaf-disease-detection
### Neem Leaf Disease Classification using Transfer Learning

7-class disease classifier for neem leaves. Compared 4 CNN architectures.
Custom EfficientNetV2B0 hit 92.62% — 14 points ahead of the next best model.

---

## Why this exists

Farmers lose 30%+ of neem crops to diseases like Alternaria, Leaf Rust, and Powdery Mildew.
Manual diagnosis is slow, expensive, and needs experts rural farmers don't have access to.
This system takes a photo of a leaf and tells you what's wrong in seconds.

---

## Dataset

Source: Mendeley Data — neem leaf image dataset
13,635 images across 7 classes — Alternaria, Dieback, Leaf Blight, Leaf Miners, Leaf Rust, Powdery Mildew, Healthy

Split: 70% train / 15% val / 15% test
Augmented minority classes to balance — 15,372 training images after augmentation

---

## The core finding

Standard transfer learning freezes pretrained weights and only trains the head.
That gave us 75%. Not good enough.

The custom model unfreezes the top 20% of EfficientNetV2B0 layers, forcing the network
to relearn domain-specific features — neem leaf texture, lesion patterns, pigmentation changes.
That gave us 92.62%.

Fine-tuning the right layers matters more than the architecture itself.

---

## Model Comparison

| Model | Test Accuracy | F1-Score |
|---|---|---|
| Xception | 74.98% | 0.750 |
| EfficientNetV2B0 | 75.61% | 0.766 |
| ResNet50V2 | 78.64% | 0.795 |
| **Custom EfficientNetV2B0** | **92.62%** | **0.930** |

---

## Custom Model Details

- Base: EfficientNetV2B0 pretrained on ImageNet
- Top 20% of layers unfrozen for domain fine-tuning
- Lightweight classification head — optimized for low-latency inference
- Callbacks: EarlyStopping + ModelCheckpoint
- Final val accuracy: 92.27% — val loss: 0.2167
- Input size: 224×224, batch size: 32

---

## Deployment

Deployed as a Streamlit web app.
Upload a leaf image → get disease class + confidence score instantly.

Test results:
- Dieback31.jpg → Predicted: Dieback, Confidence: 100%
- Leaf_Blight142.jpg → Predicted: Leaf Blight, Confidence: 63.25%
- Healthy33.jpg → Predicted: Healthy, Confidence: 84.32%

---

## Stack

Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Streamlit
Trained on Google Colab A100 GPU

---

## Limitations

- Neem-specific — won't generalize to other crops without retraining
- Some misclassification between early-stage Powdery Mildew and Healthy
- TensorFlow Lite conversion for mobile deployment not done yet
