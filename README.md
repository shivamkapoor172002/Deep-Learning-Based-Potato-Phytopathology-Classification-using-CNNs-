# Deep-Learning-Based-Potato-Phytopathology-Classification-using-CNNs-
## Overview
![image](https://github.com/shivamkapoor172002/Deep-Learning-Based-Potato-Phytopathology-Classification-using-CNNs-/assets/92868323/05cf4196-af19-4342-bc8b-ac538a07f0e5)

This project focuses on the classification of potato diseases using a deep learning model. The dataset used for training and evaluation is credited to [Plant Village](https://www.kaggle.com/arjuntejaswi/plant-village), sourced from Kaggle.

## Dependencies

- TensorFlow
- Matplotlib
- split-folders

```bash
pip install tensorflow matplotlib split-folders
```

## Dataset

The dataset has been split into training, validation, and test sets using the `splitfolders` tool:

```bash
splitfolders --ratio 0.8 0.1 0.1 -- ./training/PlantVillage/
```

## Data Augmentation

The training data is augmented using TensorFlow's `ImageDataGenerator` for better model generalization.

## Model Architecture

The classification model is a convolutional neural network (CNN) with the following layers:

- Conv2D layers with ReLU activation
- MaxPooling2D layers
- Flatten layer
- Dense layers with ReLU and Softmax activations

## Model Training

The model is compiled using the Adam optimizer and SparseCategoricalCrossentropy loss function. It is trained for 20 epochs.

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=47,
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=6,
    verbose=1,
    epochs=20,
)
```

## Model Evaluation

The trained model achieves a test accuracy of approximately 97.45%.

## Visualization

Accuracy and loss curves are plotted for both training and validation datasets.

## Inference

A sample image is run through the trained model for inference, providing both the actual and predicted labels.

## Predict Function

A utility function `predict` is provided for making predictions on new images.

```python
def predict(model, img):
    # Function details
```

## Sample Predictions

Sample predictions are visualized on a set of test images, displaying the actual and predicted labels along with confidence levels.

## Model Saving

The trained model is saved in the h5 format for easy deployment and sharing.

```python
model.save("potatoes.h5")
```

Feel free to explore, contribute, and use the model for your own potato disease classification tasks!
