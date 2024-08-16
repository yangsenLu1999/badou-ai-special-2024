import os
photo_name_list = os.listdir('./data/image/train/')  # 返回指定的文件夹包含的文件或文件夹的名字的列表，所有图片名字字符串的形成的列表
# print(photo_name_list)
with open('data/dataset.txt', 'w') as f:
    for photo in photo_name_list:
         name = photo.split('.')
         # print(name, type(name), len(name))  # ['dog', '9990', 'jpg'] <class 'list'> 3
         if name[0] == 'cat':
            f.write(photo + ';0\n')  # cat.0.jpg;0  依次每行录入图片名称后缀 + 0/1的标签
         elif name[0] == 'dog':
            f.write(photo + ';1\n')
f.close()
