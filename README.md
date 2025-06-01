# 🛡️ Adversarial Image Classifier

This project demonstrates how image classification models can be tricked using adversarial examples. Using CIFAR-10 and the CleverHans library, we show how even small, imperceptible pixel perturbations can cause significant misclassifications in an otherwise accurate model.

## 🔍 Key Features

- Trains a simple neural network on CIFAR-10
- Generates adversarial examples using FGSM
- Evaluates and visualizes prediction failures
- Explains the implications for AI security

## 📊 Example

| Original Image | Adversarial Image |
|----------------|-------------------|
| True: cat<br>Pred: cat | Pred: ship |

## 🧪 Attack Details

- **Attack method**: Fast Gradient Sign Method (FGSM)
- **Epsilon**: 0.05
- **Result**: Model accuracy drops from ~70% to ~20% on adversarial images

## 🚧 Why This Matters

This project highlights a fundamental weakness in machine learning systems: even high-performing models are not robust by default. Adversarial testing is essential in any AI security pipeline to identify and mitigate risks before deployment.

## ▶️ Usage

```bash
pip install -r requirements.txt
python adversarial_classifier.py
