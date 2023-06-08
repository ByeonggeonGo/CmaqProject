import tensorflow as tf

class Downsample(tf.keras.layers.Layer):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.layer_stack = []

        self.layer_stack.append(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=self.initializer, use_bias=False))
        if apply_batchnorm:
            self.layer_stack.append(tf.keras.layers.BatchNormalization())
        self.layer_stack.append(tf.keras.layers.LeakyReLU())


    def call(self, inputs):
        x = inputs
        for layer_id in self.layer_stack:
            x = layer_id(x)

        return x
    

class Upsample(tf.keras.layers.Layer):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.layer_stack = []

        self.layer_stack.append(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=self.initializer,
                                    use_bias=False))
        self.layer_stack.append(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            self.layer_stack.append(tf.keras.layers.Dropout(0.5))
        self.layer_stack.append(tf.keras.layers.ReLU())
        
    def call(self, inputs):
        x = inputs
        for layer_id in self.layer_stack:
            x = layer_id(x)
        return x
    

class CBR2d(tf.keras.layers.Layer):
    def __init__(self, out_channels,kernel_size = 3):
        super(CBR2d, self).__init__()
        self.cnn_layer = tf.keras.layers.Conv2D(out_channels, kernel_size, activation='relu', padding='same')
        self.batch_layer = tf.keras.layers.BatchNormalization()
              
    def call(self, inputs):
        cnn_feat_x = self.cnn_layer(inputs)
        batched_feat = self.batch_layer(cnn_feat_x)
        return batched_feat
    

class LassoRegression(tf.keras.layers.Layer):
    def __init__(self):
        super(LassoRegression, self).__init__()
        
        self.l1_regularizer = tf.keras.regularizers.L1(0.01)
        self.layer = tf.keras.layers.Dense(1, kernel_regularizer=self.l1_regularizer)
        
    def call(self, inputs):
        output = self.layer(inputs)
        return output
    
