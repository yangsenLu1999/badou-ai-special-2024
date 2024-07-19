import keras
from keras import layers, Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, ZeroPadding2D, Softmax
from keras.layers import Activation,BatchNormalization
from keras.models import Model
import utils

class DebugInfo:
    stage = 0
    block = None

    def __init__(self, stage, block):
        self.stage = stage
        self.block = block

class Network:
    net = None

    # 卷积
    def _Conv2D(self, input, filter, kernel, step, debugInfo):
        name = 'conv' + str(debugInfo.stage) + '_branch' + debugInfo.block
        bnName = 'bn' + str(debugInfo.stage) + '_branch' + debugInfo.block

        print("name: ", name)

        res = Conv2D(filter, kernel, strides=step, padding='SAME', name=name+'a')(input)
        res = BatchNormalization(name=bnName+'a')(res)
        res = Activation('relu')(res)
        return res

    # Depthwise Convolution/Pointwise convolution
    def _Conv2DwPw(self, input, filter, kernel, step, debugInfo):
        inChannel = int(input.shape[3])
        res = self._Conv2D(input, inChannel, kernel, step, debugInfo)
        debugInfo.block = '2'
        res = self._Conv2D(res, filter, (1, 1), 1, debugInfo)
        return res

    def __init__(self):
        self.net = Sequential()
        # # 初始化
        inputEle = Input(shape=utils.INPUT_SHAPE)
        res = ZeroPadding2D((3, 3))(inputEle)

        # 1. Conv: size = 224 * 224 * 3, kernel= 3 * 3, filter = 32, step = 2
        res = self._Conv2D(res, 32, (3, 3), 2, DebugInfo(1, '1'))
        # 2. Conv DW_PW: size = 112 * 112 * 32, kernel= 3 * 3, filter = 32, step = 1
        res = self._Conv2DwPw(res, 32, (3, 3), 1, DebugInfo(2, '1'))
        # 3. Conv: size = 112 * 112 * 32, kernel= 1 * 1, filter = 64, step = 1
        res = self._Conv2D(res, 64, (1, 1), 1, DebugInfo(3, '1'))
        # 4. Conv DW_PW: size = 112 * 112 * 64, kernel= 3 * 3, filter = 64, step = 2
        res = self._Conv2DwPw(res, 64, (3, 3), 2, DebugInfo(4, '1'))
        # 5. Conv: size = 56 * 56 * 64, kernel= 1 * 1, filter = 128, step = 1
        res = self._Conv2D(res, 128, (1, 1), 1, DebugInfo(5, '1'))
        # 6. Conv DW_PW: size = 56 * 56 * 128, kernel= 3 * 3, filter = 128, step = 1
        res = self._Conv2DwPw(res, 128, (3, 3), 1, DebugInfo(6, '1'))
        # 7. Conv: size = 56 * 56 * 128, kernel= 1 * 1, filter = 128, step = 1
        res = self._Conv2D(res, 128, (1, 1), 1, DebugInfo(7, '1'))
        # 8. Conv DW_PW: size = 56 * 56 * 128, kernel= 3 * 3, filter = 128, step = 2
        res = self._Conv2DwPw(res, 128, (3, 3), 2, DebugInfo(8, '1'))
        # 9. Conv: size = 28 * 28 * 128, kernel= 1 * 1, filter = 256, step = 1
        res = self._Conv2D(res, 256, (1, 1), 1, DebugInfo(9, '1'))
        # 10. Conv DW_PW: size = 28 * 28 * 256, kernel= 3 * 3, filter = 256, step = 1
        res = self._Conv2DwPw(res, 256, (3, 3), 1, DebugInfo(10, '1'))
        # 11. Conv: size = 28 * 28 * 256, kernel= 1 * 1, filter = 256, step = 1
        res = self._Conv2D(res, 256, (1, 1), 1, DebugInfo(11, '1'))
        # 12. Conv DW_PW: size = 28 * 28 * 256, kernel= 3 * 3, filter = 256, step = 2
        res = self._Conv2DwPw(res, 256, (3, 3), 2, DebugInfo(12, '1'))
        # 13. Conv: size = 14 * 14 * 256, kernel= 1 * 1, filter = 512, step = 1
        res = self._Conv2D(res, 512, (1, 1), 1, DebugInfo(13, '1'))

        # 14. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 1
        # 15. Conv: size = 14 * 14 * 512, kernel= 1 * 1, filter = 512, step = 1
        # 16. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 1
        # 17. Conv: size = 14 * 14 * 512, kernel= 1 * 1, filter = 512, step = 1
        # 18. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 1
        # 19. Conv: size = 14 * 14 * 512, kernel= 1 * 1, filter = 512, step = 1
        # 20. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 1
        # 21. Conv: size = 14 * 14 * 512, kernel= 1 * 1, filter = 512, step = 1
        # 22. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 1
        # 23. Conv: size = 14 * 14 * 512, kernel= 1 * 1, filter = 512, step = 1
        for i in range(14, 22, 2):
            # ConvDW_PW: size = 14 * 14 * 512, kernel = 3 * 3, filter = 512, step = 1
            res = self._Conv2DwPw(res, 512, (3, 3), 1, DebugInfo(i, '1'))
            # Conv: size = 14 * 14 * 512, kernel = 1 * 1, filter = 512, step = 1
            res = self._Conv2D(res, 512, (1, 1), 1, DebugInfo(i + 1, '1'))

        # 24. Conv DW_PW: size = 14 * 14 * 512, kernel= 3 * 3, filter = 512, step = 2
        res = self._Conv2DwPw(res, 512, (3, 3), 2, DebugInfo(24, '1'))
        # 25. Conv: size = 7 * 7 * 512, kernel= 1 * 1, filter = 1024, step = 1
        res = self._Conv2D(res, 1024, (1, 1), 1, DebugInfo(25, '1'))
        # 26. Conv DW_PW: size = 7 * 7 * 1024, kernel= 3 * 3, filter = 1024, step = 2
        res = self._Conv2DwPw(res, 1024, (3, 3), 2, DebugInfo(26, '1'))
        # 27. Conv DW_PW: size = 7 * 7 * 1024, kernel= 1 * 1, filter = 1024, step = 1
        res = self._Conv2D(res, 1024, (1, 1), 1, DebugInfo(27, '1'))


        # # 28. 池化：size = 7 * 7 * 1024, kernel= 7 * 7, step = 1
        # res = AveragePooling2D((7, 7), 1, padding='SAME')(res)
        # # 29. 全连接 FC: size = 1 * 1 * 类别数量
        # res = Flatten()(res)
        # res = Dense(utils.OUTPUT_SHAPE, activation='relu', name="FC")(res)
        # # 30. softmax：1 * 1 * 类别数量
        # res = Softmax(name="Softmax")(res)

        # # 创建模型
        self.net = Model(inputEle, res, name='model')

if __name__ == "__main__":
    myNet = Network()