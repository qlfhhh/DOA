import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense,Deconv2D,Output
#from pptx_util import save_model_to_pptx
import config


def main():
    config.inter_layer_margin = 80
    config.channel_scale = 4 / 5
    model = Model(input_shape=(16,8,8))
    model.add(Conv2D(16, (2, 2), (1, 1)))
    model.add(Conv2D(32, (2, 2), (1, 1)))
    model.add(Conv2D(64, (2, 2), (1, 1)))
    model.add(Deconv2D(32, (2, 2), (1, 1)))
    model.add(Deconv2D(16, (2, 2), (1, 1)))
    model.add(Deconv2D(1, (2, 2), (1, 1)))
    model.add(Output(pool_size=(1, 1), strides=None))
    model.save_fig(os.path.splitext(os.path.basename(__file__))[0] + ".svg")
    #save_model_to_pptx(model, os.path.splitext(os.path.basename(__file__))[0] + ".pptx")


if __name__ == '__main__':
    main()