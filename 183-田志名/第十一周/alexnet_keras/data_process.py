import os
file=os.listdir("./train/")   #读取train文件夹中的所有文件名字

with open("dataset.txt" ,"w") as f:
    for image_name in file:
        flag=image_name.split(".")[0]
        if flag=="cat":
            f.write(image_name+";0\n")
        else:
            f.write(image_name+";1\n")
f.close()