# 13-challenge
 Binary classification model using a deep neural network.

## Technologies
* import pandas as pd
* from pathlib import Path
* import tensorflow as tf
* from tensorflow.keras.layers import Dense
* from tensorflow.keras.models import Sequential
* from sklearn.model_selection import train_test_split
* from sklearn.preprocessing import StandardScaler,OneHotEncoder


## Installation Guide
Using the Conda package manager: [My GitHub Project](https://github.com/ALovettII/13-challenge.git)

You'll need to install the following tools for this project:
* TensorFlow 2.0
* Keras

**Important Note:** Apple computers using M1 chips may experience difficulty installing TensorFlow and Keras libraries. To run the project easily, adapt the code for Google Colab.

#### Apple M1 Installation:
1. Begin by following steps 1-3, described in this guide for MacOS:
* [Install TensorFlow with Pip](https://www.tensorflow.org/install/pip#macos)

2. Then follow this guide from Apple to install the PluggableDevice
* [Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

#### Non-M1 Installation:
```python
# Use the pip install command to install the TensorFlow 2.0 library
pip install --upgrade tensorflow

# Verify that the installation completed successfully.
python -c "import tensorflow as tf;print(tf.__version__)"

# Keras is included with TensorFlow 2.0 so verify package availablity
python -c "import tensorflow as tf;print(tf.keras.__version__)"
```


## Usage
Use this project to predict whether Alphabet Soup (fictional company) funding applicants will be successful, you will create a binary classification model using a deep neural network.

The project follows this outline:
* Preprocess data for the neural network model.
* Use the model-fit-predict pattern to compile and evaluate a binary classification model.
* Optimize the model.

If you'd like to adapt the code for another project:
* Ensure you have a proper data containing success of startups
* Encode the categorical variables using `OneHotEncoder`
* Optimize the model by:
    * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
    * Add more neurons (nodes) to a hidden layer.
    * Add more hidden layers.
    * Use different activation functions for the hidden layers.
    * Add or reduce the number of epochs in the training regimen.

Successful deployment should yield the following results:
![]()
![]()
![]()

To import any of the models for manipulation without compile (replacing 'model' with any one of the three models):
```python
# Importing necessary libraries
import tensorflow as tf

# Defining the filepath
file_path = Path("Resources/model.h5")

# Loading the model
nn_imported = tf.keras.models.load_model(file_path)
```


## Summary of Analysis
| Model | Hidden Layers | Epochs | Loss | Accuracy |
| ----- | ------------- | ------ | ---- | -------- |
| Original Model | 2 HL | 50 epochs | 0.5548 | 0.7306 |
| Alternative 1 | 2 HL | 70 epochs | 0.5539 | 0.7275 |
| Alternative 2 | 3 HL | 50 epochs | 0.5539 | 0.7310 | 


## Contributors
Created by Arthur Lovett