
# coding: utf-8


import os
import cv2

data_folder_path = "data/"
train_folder_name = "Train"
test_folder_name = "Test"
# Train folder exists
os.path.isdir(f"{data_folder_path}{train_folder_name}")
# Test folder exists
os.path.isdir(f"{data_folder_path}{test_folder_name}")



def extractFrames(pathIn, pathOut):
    #os.mkdir(pathOut)
 
    cap = cv2.VideoCapture(pathIn)
    count = 0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)
            #cv2.imwrite(f"{pathOut}frame{count}.jpg",frame)
            # save frame as JPEG file
            count += 1
        else:
            break
 


for root,dirs,files in os.walk(data_folder_path,topdown=True):
    if os.path.isdir(root):
        for file in files:
            if file.endswith(".mov"):
                folder_name = file.split(sep=".")[0]
                if not os.path.isdir(f"{root}/{folder_name}"):
                    os.mkdir(f"{root}/{folder_name}")
                extractFrames(f"{root}/{file}",f"{root}/{folder_name}")
                
        print("done")



