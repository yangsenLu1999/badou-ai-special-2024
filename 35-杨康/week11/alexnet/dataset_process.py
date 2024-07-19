import os

pictures = os.listdir('./data/image/train/')
with open('./data/dataset.txt','w') as f:
    for pic in pictures:
        name = pic.split('.')[0]
        if name == 'cat':
            f.write(pic + ';0' + '\n')
        elif name == 'dog':
            f.write(pic + ';1'+ '\n')
f.close()