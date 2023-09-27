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

from CVAE import *
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Sample argparse script.")
parser.add_argument("-v", "--verbose", help="Enable verbose mode.", action="store_true")
parser.add_argument("-i", "--input", type=str, help="Input file path.", default="train")
parser.add_argument("-o", "--output", type=str, help="Output file path. Should be in the .h5 file format.", default='trained_model.h5')
parser.add_argument("-e", "--epochs", help="Number of epochs", default=30)
parser.add_argument("-b", "--batch_size", help="Batch Size for SGD", default=10)

args = parser.parse_args()
TRAIN_DATASET_PATH  = args.input
MODEL_OUTPUT_PATH   = args.output
NUM_EPOCHS          = int(args.epochs)
BATCH_SIZE          = int(args.batch_size)

if args.verbose:
    print(f"Verbose mode enabled.")
    print(f"Input file: {TRAIN_DATASET_PATH}")
    print(f"Output file: {MODEL_OUTPUT_PATH}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE} ")



def plot_training(history):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot training and validation accuracy
    # plt.figure(figsize=(8, 6))
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.savefig("figures/training_val_acc.png")

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("figures/loss.png")



cvae = CVAE()

rainy_images, clean_images = load_dataset(TRAIN_DATASET_PATH)

rain_train, rain_val, clean_train, clean_val = train_test_split(rainy_images, clean_images, test_size=0.2, random_state=42)

# optimization method
optimizer = tf.keras.optimizers.Nadam()

# compile 
cvae.compile(optimizer=optimizer, run_eagerly=True)

# train
history = cvae.fit((rain_train, clean_train), 
                   batch_size=BATCH_SIZE, 
                   epochs=NUM_EPOCHS, 
                   validation_data=(rain_val, clean_val))

cvae.save_weights(MODEL_OUTPUT_PATH)

# print(history.history.keys())

plot_training(history)



