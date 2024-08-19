#2.数据标注，猫后面加0
# 这段代码用于遍历一个目录中的图片文件，并根据文件名的特定前缀（"cat"或"dog"）将文件名和相应的标签写入到一个文本文件中。
import os  # 与操作系统交互的标准库，提供了与文件和目录相关的功能。

photos = os.listdir("./data/image/train/")
# 使用 os.listdir() 方法获取指定目录（"./data/image/train/"）中的所有文件和子目录，返回一个包含所有项名称的列表

# 使用 with 语句打开一个文本文件（data/dataset.txt）用于写入（"w"表示写入模式）
# with 语句确保在代码块执行完后自动关闭文件，f 是文件对象
with open("data/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        '''将文件名通过 . 分隔符进行切分，并取切分后的第一个部分（即去掉文件扩展名的文件名）。
        例如，对于 "cat.jpg"，name 将会是 "cat"'''
        if name=="cat":
            f.write(photo + ";0\n")
        '''如果 name 等于 "cat"，则将当前文件名 photo 和对应的标签（0表示猫）写入到文件 dataset.txt 中，格式为 photo;0
        ，并在行末添加换行符 \n'''
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()