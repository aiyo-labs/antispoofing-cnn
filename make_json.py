
import json
import os

data_folder_path = "data/Val/"
train_folder_name = "Train"
test_folder_name = "Test"
# Train folder exists
os.path.isdir(f"{data_folder_path}{train_folder_name}")
# Test folder exists
os.path.isdir(f"{data_folder_path}{test_folder_name}")

train = {}
for root,dirs,files in os.walk(data_folder_path,topdown=True):
    if os.path.isdir(root):
       for file in files:
           if file.endswith(".jpg"):
               #print(f"{root}/{file}")
               file_name = f"{root}/{file}"
               file_name_depth = f"{root}/{file}_depth.jpg"
               train[file_name] = file_name_depth

train_json = json.dumps(train)
f = open("val.json","w")
f.write(train_json)
f.close()
               


