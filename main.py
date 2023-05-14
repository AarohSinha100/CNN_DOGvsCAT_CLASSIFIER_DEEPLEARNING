import tkinter
from tkinter import*
from tkinter import filedialog
import tensorflow as tf
import io


class_names = ["DOG","CAT"]

def preprocess_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[224,224])
    img = img / 255.

    return img


def getImage():
    file_path = filedialog.askopenfilename()
    model = tf.keras.models.load_model("Dogs_vs_Cats.h5") #Loading the model
    image_to_be_predicted = preprocess_image(file_path) #The image to be predicted
    pred = model.predict(tf.expand_dims(image_to_be_predicted, axis=0)) #image gets predicted

    pred_class = class_names[int(tf.round(pred))] #The class of image
    img_label.config(text=pred_class)
    print(pred_class)




window = Tk()

window.geometry("450x375")
window.title("DOGS vs CATS (CNN CLASSIFIER - DEEP LEARNING)")

label_re = tkinter.Label(text="DOGS VS CATS PREDICTION",font=('Arial',20),height=4)
img_label = tkinter.Label(text="SELECT IMAGE",width=45,height=5,relief="solid",borderwidth=2)
label_se = tkinter.Label()
upload =tkinter.Button(text="PREDICT",command=getImage,bg="#7936f5",fg="white",width=14,height=2,font=('Arial',9),activeforeground="#7936f5",activebackground="white")

label_re.pack()
img_label.pack()
label_se.pack()
upload.pack()

window.mainloop()








