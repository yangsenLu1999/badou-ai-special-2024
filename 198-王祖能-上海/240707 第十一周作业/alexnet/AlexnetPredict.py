'''
如果出现'str' object has no attribute 'decode'错误用下列方法解决
pip uninstall h5py后，执行命令pip install h5py==2.10重新安装h5py包
'''

from keras import backend as K
import numpy as np
import AlexnetUtils
import AlexnetModel
import cv2

if __name__ == '__main__':
	model = AlexnetModel.Alexnet()
	model.load_weights('logs/last1.h5')
	source = cv2.imread('Test.jpg')
	img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
	img = img / 255
	img = np.expand_dims(img, 0)
	img = AlexnetUtils.resize_img(img, (224, 224))

	argmax = np.argmax(model.predict(img))
	synset = AlexnetUtils.print_answer(argmax)
	print(synset)
	cv2.imshow(synset, source)
	cv2.waitKey(0)
