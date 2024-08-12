from keras.layers import Input, Conv2D, Reshape, TimeDistributed, Flatten, Dense
from keras.models import Model

from faster_rcnn.nets.resnet50 import create_model, classifier_layers
from faster_rcnn.nets.roipooling import RoiPolling


def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv_1')(base_layers)
    x_class = Conv2D(num_anchors, 1, activation='sigmoid', kernel_initializer='uniform', name='rnp_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, 1, activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = Reshape((-1, 1), name='rpn_classification')(x_class)
    x_regr = Reshape((-1, 4), name='rpn_regression')(x_regr)
    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPolling(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, training=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name="dense_class_{}".format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero',
                                     name="dens_regress_{}".format(nb_classes)))(out)
    return [out_class, out_regr]


def get_model(config, num_classes):
    inputs = Input((None, None, 3))
    roi_inputs = Input((None, 4))
    base_layers = create_model(inputs)

    # 初始检测框个数
    num_anchor = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchor)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_inputs, config.num_rois, nb_classes=num_classes)
    model_classifier = Model([inputs, roi_inputs], classifier)
    model_all = Model([inputs, roi_inputs], rpn[:2] + classifier)
    return model_rpn, model_classifier, model_all


def get_predict_model(config, num_classes):
    inputs = Input((None, None, 3))
    roi_input = Input((None, 4))
    feature_map_input = Input((None, None, 1024))

    base_layers = create_model(inputs)
    num_anchors = len(config.anchor_box_ratios) * len(config.anchor_box_scales)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only





