import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, Xception, MobileNetV2, EfficientNetB0, EfficientNetB1, DenseNet169
)


def model_factory(model_name, input_shape=(128, 128, 1)):
    """
    Create a specified pre-trained model adapted for single-channel input.

    Args:
    - model_name (str): Name of the pre-trained model to use.
    - input_shape (tuple): Shape of the input images (should have one channel).
    - num_classes (int): Number of output classes.

    Returns:
    - model (tf.keras.Model): Compiled model ready for training.
    """
    if model_name == 'VGG16':
        base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet169':
        base_model = DenseNet169(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'Xception':
        base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    elif model_name == 'EfficientNetB1':
        base_model = EfficientNetB1(weights=None, include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model; please choose from 'VGG16', 'ResNet50', 'InceptionV3', 'Xception', 'MobileNetV2', 'EfficientNetB0', 'EfficientNetB1', 'simple_cnn'.")

    # Create the model
    x = base_model.output
    x = Flatten()(x)  # Flatten the output layer to 1 dimension
    x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
    predictions = Dense(1, activation='relu')(x)  # Add a final output layer

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', tf.keras.metrics.RootMeanSquaredError()])
    return model
