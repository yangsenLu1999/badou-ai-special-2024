from keras import backend
from keras.engine.base_layer import Layer

if backend.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPolling(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = backend.image_data_format()
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPolling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, *args, **kwargs):
        assert (len(x) == 2)
        img = x[0]
        rois = x[1]
        outputs = []
        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = backend.cast(x, 'int32')
            y = backend.cast(y, 'int32')
            w = backend.cast(w, 'int32')
            h = backend.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = backend.concatenate(outputs, axis=0)
        final_output = backend.reshape(final_output,
                                       (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))
