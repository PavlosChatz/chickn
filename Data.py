import numpy as np
import pandas as pd
import cv2 
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import os

def train_validation_split(X, Y, train_percent):
    val_border = np.floor(train_percent * X.shape[0]).astype(int) 
    X_train = X[:val_border, :, :, :]
    Y_train = Y[:val_border]
    X_validate = X[val_border:, :, :, :]
    Y_validate = Y[val_border: ]
    return X_train, Y_train, X_validate, Y_validate

def load_data_animals10(dims = (224, 224), base_path = ".",):
    # dims -> (h, w) 2 int tuple - output images h x w x 3
    path = base_path + "\\datasets\\animals10\\raw-img"
    foldernames = os.listdir(path)
    
    categories = []
    files = []
    images = []
    for category, folder in enumerate(foldernames):
        path_to_file = os.path.join(path, folder)
        for fname in os.listdir(path_to_file):
            files.append(fname)
            categories.append(category)
            #img = Image.open(os.path.join(path_to_file, fname)
            img = cv2.imread(os.path.join(path_to_file, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB conversion
            img = cv2.resize(img, dims)  # resize to desired dimensions
            images.append(img)

    df = pd.DataFrame({"filenames": files, "categories": categories})
    images, categories, files = shuffle(images, categories, files)    
    X = np.array(images) / 255 
    Y = np.array(categories)
    Y = to_categorical(Y, num_classes = 10)
    X_train, Y_train, X_val, Y_val = train_validation_split(X, Y, 0.7)
    return X_train, Y_train, X_val, Y_val
