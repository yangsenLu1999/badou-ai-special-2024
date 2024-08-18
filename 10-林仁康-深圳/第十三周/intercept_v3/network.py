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

        res = Conv2D(filter, kernel, strides=step, padding='SAME', name=name+'a')(input)
        res = BatchNormalization(name=bnName+'a')(res)
        res = Activation('relu')(res)
        return res

    def _Inception1And2And3(self, input, stage):
        # 1 channel = 64, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 64, (1, 1), 1, DebugInfo(stage, '1'))

        # 2.1 channel = 48, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 48, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 64, kernel= 5 * 5, step = 1
        res2 = self._Conv2D(res2, 64, (5, 5), 1, DebugInfo(stage, '2.2'))

        # 3.1 channel = 64, kernel= 1 * 1, step = 1
        res3 = self._Conv2D(input, 64, (1, 1), 1, DebugInfo(stage, '3.1'))
        # 3.2 channel = 96, kernel= 3 * 3, step = 1
        res3 = self._Conv2D(res3, 96, (3, 3), 1, DebugInfo(stage, '3.2'))
        # 3.3 channel = 96, kernel= 3 * 3, step = 1
        res3 = self._Conv2D(res3, 96, (3, 3), 1, DebugInfo(stage, '3.3'))

        # 4.1 avg pool: kernel= 3 * 3, step = 1
        res4 = AveragePooling2D((3, 3), 1, padding='SAME')(input)
        # 4.2 channel = 32, kernel= 1 * 1, step = 1
        res4 = self._Conv2D(res4, 32, (1, 1), 1, DebugInfo(stage, '3.4'))

        # 融合
        layers.concatenate([res1, res2, res3, res4])

    def _Inception2_1(self, input, stage):
        # 1 channel = 384, kernel= 3 * 3, step = 2
        res1 = self._Conv2D(input, 384, (3, 3), 2, DebugInfo(stage, '1'))

        # 2.1 channel = 64, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 64, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 96, kernel= 3 * 3, step = 1
        res2 = self._Conv2D(res2, 96, (3, 3), 1, DebugInfo(stage, '2.2'))
        # 2.3 channel = 96, kernel= 3 * 3, step = 2
        res2 = self._Conv2D(res2, 96, (3, 3), 2, DebugInfo(stage, '2.3'))

        # 3.1 max pool: kernel= 3 * 3, step = 2
        res3 = MaxPooling2D((3, 3), 2, padding='SAME')(input)

        # 融合
        layers.concatenate([res1, res2, res3])
        pass

    def _Inception2_2(self, input, stage):
        # 1 channel = 192, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '1'))

        # 2.1 channel = 128, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 128, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 128, kernel= 1 * 7, step = 1
        res2 = self._Conv2D(res2, 128, (1, 7), 1, DebugInfo(stage, '2.2'))
        # 2.3 channel = 192, kernel= 7 * 1, step = 1
        res2 = self._Conv2D(res2, 128, (7, 1), 1, DebugInfo(stage, '2.3'))

        # 3.1 channel = 128, kernel= 1 * 1, step = 1
        res3 = self._Conv2D(input, 128, (1, 1), 1, DebugInfo(stage, '3.1'))
        # 3.2 channel = 128, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 128, (7, 1), 1, DebugInfo(stage, '3.2'))
        # 3.3 channel = 128, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 128, (1, 7), 1, DebugInfo(stage, '3.3'))
        # 3.4 channel = 128, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 128, (7, 1), 1, DebugInfo(stage, '3.4'))
        # 3.5 channel = 192, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 192, (1, 7), 1, DebugInfo(stage, '3.5'))

        # 4.1 avg pool: kernel= 3 * 3, step = 1
        res4 = AveragePooling2D((3, 3,), 1, padding='SAME')(input)
        # 4.2 channel = 192, kernel= 1 * 1, step = 1
        res4 = self._Conv2D(res4, 192, (1, 1), 1, DebugInfo(stage, '4'))

        # 融合
        layers.concatenate([res1, res2, res3, res4])

    def _Inception2_3And4(self, input, stage):
        # 1 channel = 192, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '1'))

        # 2.1 channel = 160, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 160, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 160, kernel= 1 * 7, step = 1
        res2 = self._Conv2D(res2, 160, (1, 7), 1, DebugInfo(stage, '2.2'))
        # 2.3 channel = 192, kernel= 7 * 1, step = 1
        res2 = self._Conv2D(res2, 192, (7, 1), 1, DebugInfo(stage, '2.3'))

        # 3.1 channel = 160, kernel= 1 * 1, step = 1
        res3 = self._Conv2D(input, 160, (1, 1), 1, DebugInfo(stage, '3.1'))
        # 3.2 channel = 160, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 160, (7, 1), 1, DebugInfo(stage, '3.2'))
        # 3.3 channel = 160, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 160, (1, 7), 1, DebugInfo(stage, '3.3'))
        # 3.4 channel = 160, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 160, (7, 1), 1, DebugInfo(stage, '3.4'))
        # 3.5 channel = 192, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 192, (1, 7), 1, DebugInfo(stage, '3.5'))

        # 4.1 avg pool: kernel= 3 * 3, step = 1
        res4 = AveragePooling2D((3, 3), 1, padding='SAME')(input)
        # 4.2 channel = 192, kernel= 1 * 1, step = 1
        res4 = self._Conv2D(res4, 192, (1, 1), 1, DebugInfo(stage, '4'))

        # 融合
        layers.concatenate([res1, res2, res3, res4])
        pass

    def _Inception2_5(self, input, stage):
        # 1 channel = 192, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '1'))

        # 2.1 channel = 192, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 192, kernel= 1 * 7, step = 1
        res2 = self._Conv2D(res2, 192, (1, 7), 1, DebugInfo(stage, '2.2'))
        # 2.3 channel = 192, kernel= 7 * 1, step = 1
        res2 = self._Conv2D(res2, 192, (7, 1), 1, DebugInfo(stage, '2.3'))

        # 3.1 channel = 192, kernel= 1 * 1, step = 1
        res3 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '3.1'))
        # 3.2 channel = 192, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 192, (7, 1), 1, DebugInfo(stage, '3.2'))
        # 3.3 channel = 192, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 192, (1, 7), 1, DebugInfo(stage, '3.3'))
        # 3.4 channel = 192, kernel= 7 * 1, step = 1
        res3 = self._Conv2D(res3, 192, (7, 1), 1, DebugInfo(stage, '3.4'))
        # 3.5 channel = 192, kernel= 1 * 7, step = 1
        res3 = self._Conv2D(res3, 192, (1, 7), 1, DebugInfo(stage, '3.5'))

        # 4.1 avg pool: kernel= 3 * 3, step = 1
        res4 = AveragePooling2D((3, 3), 1, padding='SAME')(input)
        # 4.2 channel = 192, kernel= 1 * 1, step = 1
        res4 = self._Conv2D(res4, 192, (1, 1), 1, DebugInfo(stage, '4'))

        # 融合
        layers.concatenate([res1, res2, res3, res4])

    def _Inception3_1(self, input, stage):
        # 1.1 channel = 192, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '1.1'))
        # 1.2 channel = 320, kernel= 3 * 3, step = 2
        res1 = self._Conv2D(res1, 320, (3, 3), 2, DebugInfo(stage, '1.2'))

        # 2.1 channel = 192, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 192, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2 channel = 192, kernel= 1 * 7, step = 1
        res2 = self._Conv2D(res2, 192, (1, 7), 1, DebugInfo(stage, '2.2'))
        # 2.3 channel = 192, kernel= 7 * 1, step = 1
        res2 = self._Conv2D(res2, 192, (7, 1), 1, DebugInfo(stage, '2.3'))
        # 2.4 channel = 192, kernel= 3 * 3, step = 2
        res2 = self._Conv2D(res2, 192, (3, 3), 2, DebugInfo(stage, '2.4'))

        # 3.1 max pool: kernel= 3 * 3, step = 2
        res3 = MaxPooling2D((3, 3), 2, padding='SAME')(input)

        # 融合
        layers.concatenate([res1, res2, res3])

    def _Inception3_2And3(self, input, stage):
        # 1 channel = 320, kernel= 1 * 1, step = 1
        res1 = self._Conv2D(input, 320, (1, 1), 1, DebugInfo(stage, '1.1'))

        # 2.1 channel = 384, kernel= 1 * 1, step = 1
        res2 = self._Conv2D(input, 384, (1, 1), 1, DebugInfo(stage, '2.1'))
        # 2.2.1 channel = 384, kernel= 1 * 3, step = 1
        res2_2_1 = self._Conv2D(res2, 384, (1, 3), 1, DebugInfo(stage, '2.2.1'))
        # 2.2.2 channel = 192, kernel= 3 * 1, step = 1
        res2_2_2 = self._Conv2D(res2, 384, (3, 1), 1, DebugInfo(stage, '2.2.2'))
        res2 = layers.concatenate([res2_2_1, res2_2_2])

        # 3.1 channel = 448, kernel= 1 * 1, step = 1
        res3 = self._Conv2D(input, 448, (1, 1), 1, DebugInfo(stage, '3.1'))
        # 3.2 channel = 384, kernel= 3 * 3, step = 1
        res3 = self._Conv2D(res3, 384, (1, 1), 1, DebugInfo(stage, '3.2'))
        # 3.4.1 channel = 384, kernel= 1 * 3, step = 1
        res_3_3_1 = self._Conv2D(res3, 384, (1, 3), 1, DebugInfo(stage, '3.2.1'))
        # 3.4.2 channel = 384, kernel= 3 * 1, step = 1
        res_3_3_2 = self._Conv2D(res3, 384, (3, 1), 1, DebugInfo(stage, '3.2.1'))
        res3 = layers.concatenate([res_3_3_1, res_3_3_2])

        # 4.1 avg pool: kernel= 3 * 3, step = 1
        res4 = AveragePooling2D((3, 3), 1, padding='SAME')(input)
        # 4.2 channel = 192, kernel= 1 * 1, step = 1
        res4 = self._Conv2D(res4, 192, (1, 1), 1, DebugInfo(stage, '4'))

        # 融合
        layers.concatenate([res1, res2, res3, res4])

    def __init__(self):
        self.net = Sequential()
        # # 初始化
        inputEle = Input(shape=utils.INPUT_SHAPE)
        res = ZeroPadding2D((3, 3))(inputEle)

        # 1. size = 299 * 299 * 3, kernel= 3 * 3, step = 2
        res = self._Conv2D(res, 32, (3, 3), 2, DebugInfo(1, '1'))
        # 2. size = 149 * 149 * 32, kernel= 3 * 3, step = 1
        res = self._Conv2D(res, 32, (3, 3), 1, DebugInfo(2, '1'))
        # 3. size = 147 * 147 * 32, kernel= 3 * 3, step = 1
        res = self._Conv2D(res, 64, (3, 3), 1, DebugInfo(3, '1'))
        # 4. 池化：size = 147 * 147 * 64, kernel= 3 * 3, step = 2
        res = MaxPooling2D((3, 3), 2, padding='SAME')(res)
        # 5. size = 73 * 73 * 64, kernel= 3 * 3, step = 1
        res = self._Conv2D(res, 80, (3, 3), 1, DebugInfo(4, '1'))
        # 6. size = 71 * 71 * 80, kernel= 3 * 3, step = 2
        res = self._Conv2D(res, 192, (3, 3), 2, DebugInfo(5, '1'))
        # 7. 池化：size = 35* 35 * 192, kernel= 3 * 3, step = 1
        res = MaxPooling2D((3, 3), 1, padding='SAME')(res)
        # 8. inception 1: size = 35* 35 * 288
        self._Inception1And2And3(res, 8)
        self._Inception1And2And3(res, 9)
        self._Inception1And2And3(res, 10)
        # 9. inception 2: size = 17* 17 * 768
        self._Inception2_1(res, 11)
        self._Inception2_2(res, 12)
        self._Inception2_3And4(res, 13)
        self._Inception2_3And4(res, 14)
        self._Inception2_5(res, 15)
        # 10. inception 3: size = 8* 8 * 1280
        self._Inception3_1(res, 16)
        self._Inception3_2And3(res, 16)
        self._Inception3_2And3(res, 16)
        # 11. 池化：size = 8 * 8 * 2048, kernel= 8 * 8, step = 1
        res = MaxPooling2D((8, 8), 1, padding='SAME')(res)
        # 12. 全连接 FC: size = 1 * 1 * 类别数量
        res = Flatten()(res)
        res = Dense(utils.OUTPUT_SHAPE, activation='relu', name="FC")(res)
        # 13. softmax：1 * 1 * 类别数量
        res = Softmax(name="Softmax")(res)

        # # 创建模型
        self.net = Model(inputEle, res, name='model')

if __name__ == "__main__":
    myNet = Network()