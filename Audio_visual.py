import os
import random
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from imutils import paths
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import cv2
import sys

class Audio_visual:

#trainsfer the labels to one-hot label
    def __init__(self,path):
        self.path = path
        self.Y = os.listdir(self.path)
        #print(Y)
        self.label_encoder = LabelEncoder()
        integer_result = self.label_encoder.fit(self.Y)
        integer_result = self.label_encoder.transform(self.Y)
        self.one_hot_encoder = OneHotEncoder()
        integer_result1 = integer_result.reshape(len(integer_result), 1)
        #integer_result1 = integer_result1.flatten()
        self.classes = self.one_hot_encoder.fit(integer_result1)

        self.classes = self.one_hot_encoder.transform(integer_result1)

        #print(self.classes.toarray().shape)

#return the original labels and the one-hot one

    def get_class(self):
        return self.Y,self.classes

#generate the list of paths
    def generate_file_list(self,p):
        print("[INFO] loading images...")
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
        random.seed(42)
        random.shuffle(imagePaths)

        return imagePaths
        
#generate batches for training
    def generate_array(self,path,batch_size=32):  
        while 1:  
            random.shuffle(path)
            f = path
            cnt = 0  
            X1 = []  
            X2 = []
            Y =[]  
            for line in f:  
                image_path = sorted(list(paths.list_images(line)))
                images=[]
                for img in image_path:
                    # load the image, pre-process it, and store it in the data list
                    image = cv2.imread(img)
                    image = img_to_array(image)
                    image = np.reshape(image,(120,120,3))
                    images.append(image)
                    
                images = np.array(images,dtype="float")
                #print(images.shape)
                if(len(images)==0):
                    print(line)
                    break
                #images=np.reshape(images,(120,120,29))
                X1.append(images)
                basename = os.path.basename(line)
                
                audio_path = 'audio/'
                read_path = os.path.join(audio_path,line)
                read_path = read_path+'.wav'
                (rate,sig) = wav.read(read_path)
                sig = np.reshape(sig,(-1,1))
                mfcc_feat = mfcc(sig,rate,numcep=26)
                mfcc_feat = np.reshape(mfcc_feat,(26,121))
                X2.append(mfcc_feat)

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
                    X1 = np.array(X1, dtype="float")
                    X2 = np.array(X2, dtype="float")
                    #print(X.shape)
                    yield ([X1, X2], Y)  
                    X1 = [] 
                    X2 = [] 
                    Y = []  
        f.close() 
        
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