# Flower Recognition

A deep learning project for **classifying flowers** into 5 categories: `daisy`, `dandelion`, `roses`, `sunflowers`, and `tulips` using **TensorFlow** and **Keras**.

---

## Dataset

**[Flower Photos Dataset]([https://www.tensorflow.org/datasets/catalog/flower_photos](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition))**
Contains ~4,242 images across 5 classes.

---

## Features

* **Custom CNN** with data augmentation
* **Transfer Learning** using MobileNetV2 (pre-trained on ImageNet)
* Optimized data pipeline with `tf.data`

---

## Usage

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(224,224), batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(224,224), batch_size=32)

# Train model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
```

---

## Results

* **Validation Accuracy:** ~89% (MobileNetV2)
* Plots for training & validation accuracy and loss included

