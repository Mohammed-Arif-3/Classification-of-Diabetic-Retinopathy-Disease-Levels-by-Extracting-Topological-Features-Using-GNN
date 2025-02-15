from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image

main = tkinter.Tk()
main.title("Diabetic Retinopathy Prediction")  # Designing main screen
main.geometry("1000x650")

global filename, X, Y, cnn_model
global train_batches, valid_batches, test_batches


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % rows == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    plt.show()


def loadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")


def processImage():
    global filename, X, Y
    global train_batches, valid_batches, test_batches
    train_path = 'Dataset/train'
    valid_path = 'Dataset/valid'
    test_path = 'Dataset/test'

    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['dr', 'nodr'],
                                                             batch_size=10)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['dr', 'nodr'],
                                                             batch_size=10)
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['dr', 'nodr'],
                                                            batch_size=10)
    text.insert(END, "Preprocessing completed\n\n")
    text.insert(END, "Found images belonging to 2 classes.\n\n")
    imgs, labels = next(train_batches)
    plots(imgs, titles=labels)


def plot_confusion_matrix(cm, classes, normalize=False, title='CNN Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def runCNN():
    global train_batches, valid_batches, test_batches, cnn_model
    text.delete('1.0', END)

    # Define the model architecture
    vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(224, 224, 3))
    cnn_model = Sequential()

    for layer in vgg16_model.layers:
        cnn_model.add(layer)

    for layer in cnn_model.layers:
        layer.trainable = False

    cnn_model.add(Flatten())
    cnn_model.add(Dense(2, activation='softmax'))

    # Use 'learning_rate' instead of 'lr'
    cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Check if model weights exist and load them
    if os.path.exists('diabetic_retinopathy.h5'):
        cnn_model.tf.keras.models.load_model("diabetic_retinopathy.h5")
    else:
        cnn_model.fit(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=12,
                      verbose=2)
        cnn_model.save('diabetic_retinopathy.h5')

    text.insert(END, "CNN training process completed\n")
    text.update_idletasks()

    predictions = cnn_model.predict(test_batches, steps=1, verbose=0)
    test_imgs, test_labels = next(test_batches)

    cm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1))
    cm_plot_labels = ['dr', 'nodr']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def predict():
    global cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = Image.open(filename)
    processed_image = preprocess_image(img, target_size=(224, 224))
    prediction = cnn_model.predict(processed_image).tolist()
    print(prediction)
    predict = np.argmax(prediction)
    result = "None"
    if predict == 0:
        result = "Diabetes Retinopathy Detected"
    if predict == 1:
        result = "No Diabetes Retinopathy Detected"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, result, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow(result, img)
    cv2.waitKey(0)


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color='red')
    plt.plot(accuracy, 'ro-', color='green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Diabetic Retinopathy Prediction', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100, y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Retinopathy Dataset", command=loadDataset)
uploadButton.place(x=10, y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Images", command=processImage)
processButton.place(x=300, y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run CNN Algorithm", command=runCNN)
svmButton.place(x=590, y=100)
svmButton.config(font=font1)

predictButton = Button(main, text="Predict Retinopathy", command=predict)
predictButton.place(x=10, y=200)
predictButton.config(font=font1)

graphButton = Button(main, text="CNN Accuracy & Loss Graph", command=graph)
graphButton.place(x=300, y=200)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=120)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=300)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
