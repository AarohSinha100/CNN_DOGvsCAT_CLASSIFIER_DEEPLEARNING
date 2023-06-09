# -*- coding: utf-8 -*-
"""CAT_VS_DOG_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UV9DFQxs80-E6domu75Zk3Tf8D-_agPi

#### Upload api key from downloads everytime before restart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zip_ref = zipfile.ZipFile("/content/dogs-vs-cats.zip","r")
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D, Flatten

!ls test/cats

"""### VISUALIZATION"""

# Create class names
import pathlib
data_dir = pathlib.Path("train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

print(class_names)

"""#### Visualizing with Matplotlib"""

import random
import os
import matplotlib.image as mpimg

#Going to select image randomly
def view_random_image(target_dir, target_class):
  #Set the target directory, we will view images from here
  target_folder = target_dir + target_class

  #Get the random image path
  random_image = random.sample(os.listdir(target_folder),1)

  #Read the image and plot using matplotlib
  img = mpimg.imread(target_folder+"/"+random_image[0]) #It is going to be returned back as a list
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"The shape of the image is: {img.shape}") #Show the shape of the image

  return img

img_cat = view_random_image(target_dir="train/",
                        target_class='cats')

img_dog = view_random_image(target_dir="train/",
                            target_class='dogs')

img_dog.shape

img_cat.shape

"""### Preprocessing Our Image"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed
tf.random.set_seed(42)

# Preprocess the data (we want to get all the data between 0 and 1)
train_datagen = ImageDataGenerator(rescale=1/255.) #Rescaled Images
valid_datagen = ImageDataGenerator(rescale=1/255.) #Rescaled Images

# Setup paths to our data
train_dir = "/content/train"
test_dir = "/content/test"

# Import data from directory and turn them into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               seed=42)

len(train_data) #turned into batches

"""### Build the first CNN Model"""

#Build the CNN
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224,224,3)),
    
    #Another layer
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.Conv2D(10,3,activation="relu"),

    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

#Compile the CNN
model_1.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Fit the model
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

"""### Plotting the losses"""

def plot_loss_curves(history):
  """
  Returns seperate loss curves for training and validation metrics
  """

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"])) #How many epochs did we run for

  #Plpot accuracy
  plt.plot(epochs,loss,label="training_loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.title("LOSS")
  plt.xlabel("epochs")
  plt.legend()

  #Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="training accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("ACCURACY")
  plt.xlabel("epochs")
  plt.legend()

plot_loss_curves(history_1)

"""### Function to prepare image to pass as prediction"""

# Creating a helper function to import image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from the filename and turns it into 
  tensor and reshapes it into (img_shape, img_shape, color_channels)
  """
  #Reads in the image
  img = tf.io.read_file(filename) #read and outputs the entire contents of the filename

  #Decode the read file into your tensor
  img = tf.image.decode_image(img) #Detects the format of the image and performs the appropriate operation to convert the image
  #Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])

  #And the real images have data between 0 and 255
  #We need to scale the images between 0 and 1
  img = img/255.

  return img

dog = load_and_prep_image("dog_1.jpg")
dog

"""#### Creating a helper function to know what the class of the predicted image is and is it actually correct"""

def pred_and_plot(model, filename, class_names=class_names):
  """
  Imports an image located at filename, makes a prediction with
   model and plots the image with predicted class as the title
  """

  #Import the target image and preprocess it
  img = load_and_prep_image(filename)

  pred = model.predict(tf.expand_dims(img, axis=0))

  #Get the predicted class
  pred_class = class_names[int(tf.round(pred))]

  #Plot the image and the predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);

pred_and_plot(model_1, "dog_1.jpg")

"""### Saving this model"""

model_1.save("Dogs_vs_Cats.h5")
print("Model saved!")

"""# Data Augmentation - For better results and reduce overfitting"""

train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             shear_range=0.5,
                                             zoom_range=0.2,
                                             width_shift_range=0.5,
                                             height_shift_range=0.5,
                                             horizontal_flip=True,
                                             vertical_flip=True)

train_datagen = ImageDataGenerator(rescale=1/255.)

test_datagen = ImageDataGenerator(rescale=1/255.)

"""#### Lets write some code to visualise data augmentation"""

# Import data and augment it from directory
print("Augment training data - ")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224,224),
                                                                   batch_size=32,
                                                                   class_mode="binary",
                                                                   shuffle=True)

# Create non augmented train data batches
print("Non Augmented training Data - ")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode="binary",
                                               shuffle=True)

print("Non Augmented test Data - ")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224,224),
                                             batch_size=32,
                                             class_mode="binary")

"""#### Lets visualise some augmented data"""

#Sample data batches
images, labels = train_data.next()
augmented_images, augmented_labels = train_data_augmented.next()

##Show original images and then the augmented images

import random
random_number = random.randint(0,31)
print(f"showing image number {random_number}")
plt.imshow(images[random_number])
plt.title(f"Original images")
plt.axis(False)

plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented Image")
plt.axis(False)

