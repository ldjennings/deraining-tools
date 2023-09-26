# Run this file as a script to train the model. Modify the batch_size and epochs variables
# to tune the training process.

# Change these to wherever your dataset is located, and where you want the saved model to
# be outputted. The dataset should be in the form of:
#
#   -Dataset directory
#       -'data' directory containing images with rain artifacts
#       -'gt' directory containing ground truth clear images
#
# The program is currently set to load .png files, but that can easily be changed by 
# modifying the load_dataset() method.
TRAIN_DATASET_PATH   = 'train'

# The model should be in .h5 format, or similar.
MODEL_OUTPUT_PATH    = 'trained_model.h5'

from CVAE import *
from sklearn.model_selection import train_test_split

def plot_training(history):
    import matplotlib.pyplot as plt

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



cvae = CVAE()

rainy_images, clean_images = load_dataset(TRAIN_DATASET_PATH)

rain_train, rain_val, clean_train, clean_val = train_test_split(rainy_images, clean_images, test_size=0.2, random_state=42)

# optimization method
optimizer = tf.keras.optimizers.Nadam()

# compile 
cvae.compile(optimizer=optimizer, run_eagerly=True)

# train
batch_size = 10
epochs = 30
history = cvae.fit((rain_train, clean_train), 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_data=(rain_val, clean_val))

cvae.save_weights(MODEL_OUTPUT_PATH)

plot_training(history)



