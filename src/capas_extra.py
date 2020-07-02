from keras.layers import InputSpec, Layer
import tensorflow as tf

class ReflectionPadding2D(Layer):
    """Clase que crea una capa para la arquitectura resnet.
    Fuente: https://github.com/davidADSP/GDL_code/blob/master/models/layers/layers.py"""

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, y, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(y, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')