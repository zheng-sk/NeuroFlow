import tensorflow as tf

def upsample3d(res_increase):
    """
        Bilinear-like 3D upsampling wrapped in a Keras layer to work with KerasTensors
    """
    def _upsample(input_tensor):
        shape = tf.shape(input_tensor)
        x_size = shape[1]
        y_size = shape[2]
        z_size = shape[3]
        c_size = shape[4]

        if res_increase == 1:
            return input_tensor

        # resize y-z
        yz = tf.reshape(input_tensor, [-1, y_size, z_size, c_size])
        yz = tf.image.resize(yz, [y_size * res_increase, z_size * res_increase], method='bilinear')
        resume_b_x = tf.reshape(yz, [-1, x_size, y_size * res_increase, z_size * res_increase, c_size])

        # reorient and resize y-x
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        yx = tf.reshape(reoriented, [-1, y_size * res_increase, x_size, c_size])
        yx = tf.image.resize(yx, [y_size * res_increase, x_size * res_increase], method='bilinear')
        resume_b_z = tf.reshape(yx, [-1, z_size * res_increase, y_size * res_increase, x_size * res_increase, c_size])

        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output_tensor

    return tf.keras.layers.Lambda(_upsample, name=f'upsample3d_x{res_increase}')

def conv3d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True):
    """
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
        For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad
    """
    reg_l2 = tf.keras.regularizers.l2(5e-7)

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        # wrap tf.pad in a Keras layer to avoid KerasTensor errors
        x = tf.keras.layers.Lambda(lambda t: tf.pad(t, [[0,0],[p,p],[p,p],[p,p],[0,0]], padding))(x)
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding='VALID',
                                   kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding,
                                   kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    return x
    

def resnet_block(x, block_name='ResBlock', channel_nr=64, scale = 1, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    tmp = x + tmp * scale
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp

class SR4DFlowNet():
    def __init__(self, res_increase):
        self.res_increase = res_increase

    def build_network(self, u, v, w, u_mag, v_mag, w_mag, low_resblock=8, hi_resblock=4, channel_nr=64):
        channel_nr = 64

        speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 0.5
        pcmr = mag * speed

        phase = tf.keras.layers.concatenate([u,v,w])
        pc    = tf.keras.layers.concatenate([pcmr, mag, speed])
        
        pc = conv3d(pc,3,channel_nr, 'SYMMETRIC', 'relu')
        pc = conv3d(pc,3,channel_nr, 'SYMMETRIC', 'relu')

        phase = conv3d(phase,3,channel_nr, 'SYMMETRIC', 'relu')
        phase = conv3d(phase,3,channel_nr, 'SYMMETRIC', 'relu')

        concat_layer = tf.keras.layers.concatenate([phase, pc])
        concat_layer = conv3d(concat_layer, 1, channel_nr, 'SYMMETRIC', 'relu')
        concat_layer = conv3d(concat_layer, 3, channel_nr, 'SYMMETRIC', 'relu')
        
        # res blocks
        rb = concat_layer
        for i in range(low_resblock):
            rb = resnet_block(rb, "ResBlock", channel_nr, pad='SYMMETRIC')

        rb = upsample3d(self.res_increase)(rb)
            
        # refinement in HR
        for i in range(hi_resblock):
            rb = resnet_block(rb, "ResBlock", channel_nr, pad='SYMMETRIC')

        # 3 separate path version
        u_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        u_path = conv3d(u_path, 3, 1, 'SYMMETRIC', None)

        v_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        v_path = conv3d(v_path, 3, 1, 'SYMMETRIC', None)

        w_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        w_path = conv3d(w_path, 3, 1, 'SYMMETRIC', None)
        

        b_out = tf.keras.layers.concatenate([u_path, v_path, w_path])

        return b_out
