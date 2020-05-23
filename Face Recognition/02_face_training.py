''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Developed by Nivas V M

'''

import cv2
import numpy as np
from PIL import Image
import os
import pickle

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    id_name = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int((os.path.split(imagePath)[-1].split(".")[1]),base=36)
        id_names=(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            id_name.append(id_names)

    return faceSamples,ids,id_name

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids,id_name = getImagesAndLabels(path)

user_id_list=[]
for x in range(len(id_name)):
    if id_name[x] not in user_id_list:
        user_id_list.append(id_name[x]) 
    
print("user_id",user_id_list)
with open('user_data', 'wb') as fp:
    pickle.dump(user_id_list, fp)

recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
