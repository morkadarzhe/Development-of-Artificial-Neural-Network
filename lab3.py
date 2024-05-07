import numpy as np
import keras
from keras import layers
from keras.api.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Function to plot MNIST images
def plot_images(original_images, augmented_images, labels, num_images=6):
    plt.figure(figsize=(12, 5))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original: {}".format(int(labels[i])))
        plt.axis("off")
        
        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Augmented")
        plt.axis("off")
    # Save the figure
    plt.savefig('last_original_vs_augmented.png')
    plt.show()


#plot statistics of the models
def plot_stats(history):

    plt.figure(figsize=(12, 5))
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')


    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()

    # Save the figure
    plt.savefig('model_stats.png')

    plt.show()
    

#best and last model comparison with images
def plot_predictions_comparison(best_model, last_model, images, labels, num_images=3):
    plt.figure(figsize=(12, 8))
    num_total_images = images.shape[0]
    random_indices = np.random.choice(num_total_images, size=num_images, replace=False)
    
    # Get predictions for the best model
    best_predictions = best_model.predict(images[random_indices])
    
    # Get predictions for the last model
    last_predictions = last_model.predict(images[random_indices])
    
    for i, idx in enumerate(random_indices):
        # Plot for the best model
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
        
        # Get top 3 classes and their probabilities for the best model
        best_top_classes = np.argsort(-best_predictions[i])[:3]
        best_top_probs = best_predictions[i][best_top_classes]
        
        # Format title text for the best model
        best_title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(best_top_classes, best_top_probs)])
        
        plt.title(f'Best Model\n{best_title_text}', color='#017653')
        plt.axis("off")
        
        # Plot for the last model
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
        
        # Get top 3 classes and their probabilities for the last model
        last_top_classes = np.argsort(-last_predictions[i])[:3]
        last_top_probs = last_predictions[i][last_top_classes]
        
        # Format title text for the last model
        last_title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(last_top_classes, last_top_probs)])
        
        plt.title(f'Last Model\n{last_title_text}', color='#017653')
        plt.axis("off")
    plt.savefig('last_and_best_comparison_images.png')  
    plt.show()



########## PREPARATiON

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


########## DATA AUGMENTATION

data_augmentation_layers = [
    layers.RandomRotation(factor=0.05),  # Random rotation with a factor of 15 degrees
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Random shift by 10% of the image height and width
    layers.RandomZoom(height_factor=0.05, width_factor=0.05),  # Random zoom by 10% of the image height and width
    layers.RandomContrast(factor=0.05),  # Random contrast adjustment by 10%
    layers.GaussianNoise(stddev=0.01),  # Add Gaussian noise with standard deviation 0.01
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

x_train_augmented = data_augmentation(x_train)

# Plot the images
plot_images(x_train, x_train_augmented, y_train)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

########## BUILDING THE MODEL

best_model_filepath = "best_model.keras"
last_model_filepath = "last_model.keras"

# Define ModelCheckpoint callback to save the model with lowest validation loss
best_checkpoint = ModelCheckpoint(best_model_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
# Define ModelCheckpoint callback to save the last model
last_checkpoint = ModelCheckpoint(filepath=last_model_filepath, save_weights_only=False, verbose=0)

###################### BUILDING A MODEL
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

############### TRAINING THE MODEL
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train_augmented, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[best_checkpoint, last_checkpoint])

best_model = keras.models.load_model(best_model_filepath)
last_model = keras.models.load_model(last_model_filepath)

# Evaluate the best model on the testing dataset
score = best_model.evaluate(x_test, y_test, verbose=0)
print("Best Model Test loss:", score[0])
print("Best Model Test accuracy:", score[1])

# last model evaluation
score = last_model.evaluate(x_test, y_test, verbose=0)
print("Last Model Test loss:", score[0])
print("Last Model Test accuracy:", score[1])


# Plot statistics od the models
plot_stats(history)

#best and last model comparison with images
plot_predictions_comparison(best_model, last_model, x_test, y_test)
