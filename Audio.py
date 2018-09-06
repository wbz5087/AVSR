import os
import random
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from imutils import paths
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class Audio:

#trainsfer the labels to one-hot label
    def __init__(self,path):
        self.path = path
        self.Y = os.listdir(self.path)
        #print(Y[0])
        #print(Y)
        self.label_encoder = LabelEncoder()
        integer_result = self.label_encoder.fit(self.Y)
        integer_result = self.label_encoder.transform(self.Y)
        self.one_hot_encoder = OneHotEncoder()
        integer_result1 = integer_result.reshape(len(integer_result), 1)
        #integer_result1 = integer_result1.flatten()
        self.classes = self.one_hot_encoder.fit(integer_result1)

        self.classes = self.one_hot_encoder.transform(integer_result1)

#return the original labels and the one-hot one
    def get_class(self):
        return self.Y,self.classes

#transform the given label to the corresponding one-hot
    def get_label(self,part):
        integer_result = self.label_encoder.transform(part)
        integer_result1 = integer_result.reshape(len(integer_result), 1)
        label = self.one_hot_encoder.transform(integer_result1)

        return label.toarray()

#generate the list of paths
    def generate_audio_list(self,p):
        #print("[INFO] loading images...")
        data = []
        labels = []
        train_y = []
        # grab the image paths and randomly shuffle them

        labels = os.listdir(self.path)
        #labels.remove('.DS_Store')
        imagePaths = []
        #print(labels)
        for i, label in enumerate(labels):
            path_file=os.path.join(self.path, label)
            #path_file = os.path.join(path,'label')
            path_files = os.path.join(path_file,p)
            imagePath = os.listdir(path_files)
            #print(path_files)
            #imagePath.remove('.DS_Store')
            for img in imagePath:
                if img == '.DS_Store':
                    print('wrong')
                else:
                    image = os.path.join(path_files,img)
                    imagePaths.append(image)
                #print(path_files)
        #print(imagePaths)
        random.seed(42)
        random.shuffle(imagePaths)

        return imagePaths

#load audio with given label and the part like 'train'
    def load_audio(self,label,p):
        print("[INFO] loading images...")
        data = []
        labels = []
        train_y = []
        # grab the image paths and randomly shuffle them

        #labels.remove('.DS_Store')
        imagePaths = []
        #print(labels)
        path_file=os.path.join(self.path, label)
        #path_file = os.path.join(path,'label')
        path_files = os.path.join(path_file,p)
        imagePath = os.listdir(path_files)
        #print(path_files)
        #imagePath.remove('.DS_Store')
        for img in imagePath:
            if img == '.DS_Store':
                print('wrong')
            else:
                image = os.path.join(path_files,img)
                imagePaths.append(image)
                #print(path_files)
        #print(imagePaths)
        random.seed(42)
        random.shuffle(imagePaths)
        #print(imagePaths)
        # loop over the input images
        for imagePath in imagePaths:
            (rate,sig) = wav.read(imagePath)
            sig = np.reshape(sig,(-1,1))
            mfcc_feat = mfcc(sig,rate,numcep=26)
            mfcc_feat = np.reshape(mfcc_feat,(26,121))
            data.append(mfcc_feat)

            basename = os.path.basename(imagePath)
            label = basename.split('_')
            train_y.append(label[0])

        data = np.array(data, dtype="float")

        train_y = np.array(train_y)
        print(train_y.shape)

        integer_result = self.label_encoder.transform(train_y)
        integer_result1 = integer_result.reshape(len(integer_result), 1)
        y_label = self.one_hot_encoder.transform(integer_result1)

        return data,y_label


#generate batches for training
    def generate_arrays_from_file(self,filelist,batch_size=64):
        while 1:
            #random.shuffle(path)
            f = filelist
            cnt = 0
            X =[]
            Y =[]
            for line in f:
                (rate,sig) = wav.read(line)
                sig = np.reshape(sig,(-1,1))
                mfcc_feat = mfcc(sig,rate,numcep=26)
                mfcc_feat = np.reshape(mfcc_feat,(26,121))
                X.append(mfcc_feat)

                basename = os.path.basename(line)
                label = basename.split('_')
                Y.append(label[0])

                cnt += 1
                if cnt==batch_size:
                    cnt = 0
                    integer_result = self.label_encoder.transform(Y)
                    #print(integer_result)
                    integer_result1 = integer_result.reshape(len(integer_result), 1)
                    Y = self.one_hot_encoder.transform(integer_result1)
                    Y = Y.toarray()
                    #print(Y.shape)

                    X = np.array(X, dtype="float")
                    #print(X.shape)
                    yield (X, Y)
                    X = []
                    Y = []
        f.close()
