from tensorflow.keras.datasets import mnist

(train_img,train_label),(test_img,test_label)=mnist.load_data()

import matplotlib.pyplot as plt
a=train_img[0]
plt.imshow(a)
plt.show()

from tensorflow.keras import layers
from tensorflow.keras import models

network=models.Sequential()
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
network.add(layers.Dense(10,activation="softmax"))

network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

train_img=train_img.reshape((60000,28*28))
train_img=train_img.astype("float32")/255

test_img=test_img.reshape((10000,28*28))
test_img=test_img.astype("float32")/255

from tensorflow.keras.utils import to_categorical
print("before change:",test_label)
test_label=to_categorical(test_label)
train_label=to_categorical(train_label)
print("after change:",test_label)

network.fit(train_img,train_label,epochs=5,batch_size=128)
test_loss,test_acc=network.evaluate(test_img,test_label,verbose=1)

print(test_loss)
print("test_acc:",test_acc)


(train_img,train_label),(test_img,test_label)=mnist.load_data()
try1=test_img[5]
plt.imshow(try1)
plt.show()

test_img=test_img.reshape((10000,28*28))
res=network.predict(test_img)

for i in range(res[5].shape[0]):
    if res[5][i]==1:
        print("number is :",i)
        break
