""" Full assembly of the parts to form the complete network """
from keras import layers, Input
from keras.backend import concatenate
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, ZeroPadding2D, \
    Conv2DTranspose, Concatenate
from keras.layers import Activation,BatchNormalization
from keras.models import Model

INPUT_SHAPE = (572, 572, 1)

class UNet:
    net = None

    def __init__(self, n_channels, n_classes):
        self.net = Sequential()
        inputEle = Input(shape=INPUT_SHAPE)
        res = ZeroPadding2D((2, 2))(inputEle)

        inc = self.DoubleConv(res, n_channels, 64, "inc")
        print("inc shape: ", inc.shape)
        down1 = self.Down(inc, 64, 128, 'down1')
        print("down1 shape: ", down1.shape)
        down2 = self.Down(down1, 128, 256, 'down2')
        print("down2 shape: ", down2.shape)
        down3 = self.Down(down2, 256, 512, 'down3')
        print("down3 shape: ", down3.shape)
        down4 = self.Down(down3, 512, 512, 'down4')
        print("down4 shape: ", down4.shape)
        up1 = self.Up(down4, down3, 1024, 256, 'up1')
        up2 = self.Up(up1, down2, 512, 128, 'up2')
        up3 = self.Up(up2, down1, 256, 64, 'up3')
        up4 = self.Up(up3, inc, 128, 64, 'up4')
        outc = self.OutConv(up4, n_classes, 'output')

        # 创建模型
        self.net = Model(inputEle, outc, name='model')

    def DoubleConv(self, input, in_channels, out_channels, name):
        res = Conv2D(in_channels, (3, 3), activation='relu', padding='SAME', name=name + '_conv_a')(input)
        res = Conv2D(out_channels, (3, 3), activation='relu', padding='SAME', name=name + '_conv_b')(res)
        return res

    def Up(self, input, conv, in_channels, out_channels, name):
        # 反卷积
        res = Conv2DTranspose(in_channels//2, (2, 2), strides=(2, 2), padding='SAME', name=name+'_convT')(input)
        # 最后一维度合并
        res = Concatenate(axis=-1)([res, conv])
        return self.DoubleConv(res, in_channels, out_channels, name)

    def Down(self, input, in_channels, out_channels, name):
        res = MaxPooling2D((2, 2), name=name+'_maxpool')(input)
        return self.DoubleConv(res, in_channels, out_channels, name)

    def OutConv(self, input, out_channels, name):
        return Conv2D(out_channels, (1, 1), activation='relu', padding='SAME', name=name + '_out_conv')(input)

if __name__ == '__main__':
    UNet(1, 1)
