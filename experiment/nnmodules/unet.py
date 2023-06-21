import tensorflow as tf
from .customlayers import Upsample, Downsample, CBR2d

class Unet_v1(tf.keras.Model): 
    def __init__(self): 
        super(Unet_v1, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()

        self.down_stack = [
            Downsample(64, 3),  # (batch_size, 64, 64, 64)
            Downsample(128, 3),  # (batch_size, 32, 32, 128)
            Downsample(256, 3),  # (batch_size, 16, 16, 256)
            Downsample(512, 3),  # (batch_size, 8, 8, 512)
            Downsample(512, 3),  # (batch_size, 4, 4, 512)
        ]

        self.up_stack = [
            Upsample(512, 3, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            Upsample(512, 3),  # (batch_size, 8, 8, 1024)
            Upsample(256, 3),  # (batch_size, 16, 16, 512)
            Upsample(128, 3),  # (batch_size, 32, 32, 256)
            Upsample(64, 3),  # (batch_size, 64, 64, 128)
        ]

        self.initializer = tf.random_normal_initializer(0., 0.02)
        # self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,


        self.last_stack = [
            tf.keras.layers.Conv2DTranspose(64,4,
                                                strides=2,
                                                padding='same',
                                                # kernel_initializer=self.initializer,
                                                ),
            # tf.keras.layers.ReLU(),

            # tf.keras.layers.Conv2D(32, 3,
            #                                     # strides=2,
            #                                     padding='same',
            #                                     kernel_initializer=self.initializer,
            #                                     ),
            # tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(1, 3,
                                                strides=2,
                                                padding='same',
                                                # kernel_initializer=self.initializer,
                                                ),
                                                
                                                ]
        self.input_resize_layer = tf.keras.layers.Resizing(
            128,
            128,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )
        
    
    def call(self, input): 
        x = self.input_resize_layer(input)

        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        for last_layer in self.last_stack:
            x = last_layer(x)
        x = self.output_resize_layer(x)
        return x
    
class Unet_v2(tf.keras.Model): 
    def __init__(self,): 
        super(Unet_v2, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()

        self.input_resize_layer = tf.keras.layers.Resizing(
            128,
            128,
            interpolation='lanczos3',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='lanczos3',
            crop_to_aspect_ratio=False,
        )

        self.enc1_1 = CBR2d(out_channels = 64) 
        self.enc1_2 = CBR2d(out_channels=64)
            
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc2_1 = CBR2d(out_channels=128)
        self.enc2_2 = CBR2d(out_channels=128)
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc3_1 = CBR2d(out_channels=256)
        self.enc3_2 = CBR2d(out_channels=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )

        self.enc4_1 = CBR2d(out_channels=512)
        self.enc4_2 = CBR2d(out_channels=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        
        self.enc5_1 = CBR2d(out_channels=1024)
        self.dec5_1 = CBR2d(out_channels=512)

        self.unpool4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2,strides=(2, 2))

        self.dec4_2 = CBR2d(out_channels=512)
        self.dec4_1 = CBR2d(out_channels=256)

        self.unpool3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2,strides=(2, 2))

        self.dec3_2 = CBR2d(out_channels=256)
        self.dec3_1 = CBR2d(out_channels=128)

        self.unpool2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2,strides=(2, 2))

        self.dec2_2 = CBR2d(out_channels=128)
        self.dec2_1 = CBR2d(out_channels=64)

        self.unpool1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2,strides=(2, 2))

        self.dec1_2 = CBR2d(out_channels=64)
        self.dec1_1 = CBR2d(out_channels=64)

        self.outlayer = tf.keras.layers.Conv2D(1,kernel_size = 1)
              
    def call(self, input): 

        input = self.input_resize_layer(input)
      
        enc1_1 = self.enc1_1(input)
        enc1_2 = self.enc1_2(enc1_1)
        
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        
        pool2= self.pool2(enc2_2)
  
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        
        pool3= self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        
        pool4= self.pool3(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)
        

        unpool4 = self.unpool4(dec5_1)

        cat4 = tf.keras.layers.Concatenate(axis=3)([unpool4, enc4_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
       
        unpool3 = self.unpool3(dec4_1)
    
        cat3 = tf.keras.layers.Concatenate(axis=3)([unpool3, enc3_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        
        cat2 = tf.keras.layers.Concatenate(axis=3)([unpool2, enc2_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        
        cat1 = tf.keras.layers.Concatenate(axis=3)([unpool1, enc1_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.outlayer(dec1_1)

        x = self.output_resize_layer(x)
        return x
    
class Unet_v2_1(tf.keras.Model): 
    def __init__(self,): 
        super(Unet_v2_1, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()

        self.input_resize_layer = tf.keras.layers.Resizing(
            128,
            128,
            interpolation='lanczos3',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='lanczos3',
            crop_to_aspect_ratio=False,
        )

        self.enc1_1 = CBR2d(out_channels = 64) 
        self.enc1_2 = CBR2d(out_channels=64)
            
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc2_1 = CBR2d(out_channels=128)
        self.enc2_2 = CBR2d(out_channels=128)
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc3_1 = CBR2d(out_channels=256)
        self.dec3_1 = CBR2d(out_channels=128)

        self.unpool2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2,strides=(2, 2))

        self.dec2_2 = CBR2d(out_channels=128)
        self.dec2_1 = CBR2d(out_channels=64)

        self.unpool1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2,strides=(2, 2))

        self.dec1_2 = CBR2d(out_channels=64)
        self.dec1_1 = CBR2d(out_channels=64)

        self.outlayer = tf.keras.layers.Conv2D(1,kernel_size = 1)
              
    def call(self, input): 

        input = self.input_resize_layer(input)
      
        enc1_1 = self.enc1_1(input)
        enc1_2 = self.enc1_2(enc1_1)
        
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        
        pool2= self.pool2(enc2_2)
  
        enc3_1 = self.enc3_1(pool2)
        
        dec3_1 = self.dec3_1(enc3_1)
        
        unpool2 = self.unpool2(dec3_1)
        
        cat2 = tf.keras.layers.Concatenate(axis=3)([unpool2, enc2_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        
        cat1 = tf.keras.layers.Concatenate(axis=3)([unpool1, enc1_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.outlayer(dec1_1)

        x = self.output_resize_layer(x)
        return x
    
    
class Unet_v3(tf.keras.Model): 
    def __init__(self,base_map): 
        super(Unet_v3, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()
        self.mulp_l = tf.keras.layers.Multiply()
        self.base_map = base_map

        self.input_resize_layer = tf.keras.layers.Resizing(
            256,
            256,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.enc1_1 = CBR2d(out_channels = 64) 
        self.enc1_2 = CBR2d(out_channels=64)
            
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc2_1 = CBR2d(out_channels=128)
        self.enc2_2 = CBR2d(out_channels=128)
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc3_1 = CBR2d(out_channels=256)
        self.enc3_2 = CBR2d(out_channels=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )

        self.enc4_1 = CBR2d(out_channels=512)
        self.enc4_2 = CBR2d(out_channels=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        
        self.enc5_1 = CBR2d(out_channels=1024)
        self.dec5_1 = CBR2d(out_channels=512)

        self.unpool4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2,strides=(2, 2))

        self.dec4_2 = CBR2d(out_channels=512)
        self.dec4_1 = CBR2d(out_channels=256)

        self.unpool3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2,strides=(2, 2))

        self.dec3_2 = CBR2d(out_channels=256)
        self.dec3_1 = CBR2d(out_channels=128)

        self.unpool2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2,strides=(2, 2))

        self.dec2_2 = CBR2d(out_channels=128)
        self.dec2_1 = CBR2d(out_channels=64)

        self.unpool1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2,strides=(2, 2))

        self.dec1_2 = CBR2d(out_channels=64)
        self.dec1_1 = CBR2d(out_channels=64)

        self.outlayer = tf.keras.layers.Conv2D(1,kernel_size = 1)
              
    def call(self, input): 

       
        x = tf.multiply(tf.keras.layers.Reshape((1,1, 119))(input), self.base_map)
        x = self.input_resize_layer(x)

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        
        pool2= self.pool2(enc2_2)
  
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        
        pool3= self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        
        pool4= self.pool3(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)
        

        unpool4 = self.unpool4(dec5_1)

        cat4 = tf.keras.layers.Concatenate(axis=3)([unpool4, enc4_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
       
        unpool3 = self.unpool3(dec4_1)
    
        cat3 = tf.keras.layers.Concatenate(axis=3)([unpool3, enc3_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        
        cat2 = tf.keras.layers.Concatenate(axis=3)([unpool2, enc2_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        
        cat1 = tf.keras.layers.Concatenate(axis=3)([unpool1, enc1_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.outlayer(dec1_1)
        x = self.output_resize_layer(x)
        return x

class Unet_v4(tf.keras.Model): 
    def __init__(self,base_map): 
        super(Unet_v4, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()
        self.mulp_l = tf.keras.layers.Multiply()
        self.base_map = base_map
        self.concat_n = 32

        self.input_resize_layer = tf.keras.layers.Resizing(
            256,
            256,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.mlp_stack = [
            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(self.concat_n),

            # 그냥 인풋데이터 처리하는 부분 # 이후 베이스맵이랑 베이스스모크 cnn레이어로 처리해서 아웃풋 채널수 맞춰주고 내적한 값으로 예측하도록
        ]

        self.left_stack = [
            tf.keras.layers.Conv2D(64, 3, padding='same',use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(64, 3, padding='same',use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(self.concat_n, 3, padding='same',use_bias=True),
        ]

        self.enc1_1 = CBR2d(out_channels = 64) 
        self.enc1_2 = CBR2d(out_channels=64)
            
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc2_1 = CBR2d(out_channels=128)
        self.enc2_2 = CBR2d(out_channels=128)
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc3_1 = CBR2d(out_channels=256)
        self.enc3_2 = CBR2d(out_channels=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )

        self.enc4_1 = CBR2d(out_channels=512)
        self.enc4_2 = CBR2d(out_channels=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        
        self.enc5_1 = CBR2d(out_channels=1024)
        self.dec5_1 = CBR2d(out_channels=512)

        self.unpool4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2,strides=(2, 2))

        self.dec4_2 = CBR2d(out_channels=512)
        self.dec4_1 = CBR2d(out_channels=256)

        self.unpool3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2,strides=(2, 2))

        self.dec3_2 = CBR2d(out_channels=256)
        self.dec3_1 = CBR2d(out_channels=128)

        self.unpool2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2,strides=(2, 2))

        self.dec2_2 = CBR2d(out_channels=128)
        self.dec2_1 = CBR2d(out_channels=64)

        self.unpool1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2,strides=(2, 2))

        self.dec1_2 = CBR2d(out_channels=64)
        self.dec1_1 = CBR2d(out_channels=64)

        self.outlayer = tf.keras.layers.Conv2D(1,kernel_size = 1)
              
    def call(self, input): 

       
        # x = tf.multiply(tf.keras.layers.Reshape((1,1, 119))(input), self.base_map)
        x_left = tf.multiply(tf.ones([input.shape[0],1,1,1]), self.base_map)
        x_left = self.input_resize_layer(x_left)
        x = input

        # right_module
        for seq in self.mlp_stack:
            x = seq(x)

        for seq in self.left_stack:
            x_left = seq(x_left)
        

        x_after = tf.einsum('bijk, bk ->bijk', x_left, x)
        
        

        enc1_1 = self.enc1_1(x_after)
        enc1_2 = self.enc1_2(enc1_1)
        
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        
        pool2= self.pool2(enc2_2)
  
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        
        pool3= self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        
        pool4= self.pool3(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)
        

        unpool4 = self.unpool4(dec5_1)

        cat4 = tf.keras.layers.Concatenate(axis=3)([unpool4, enc4_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
       
        unpool3 = self.unpool3(dec4_1)
    
        cat3 = tf.keras.layers.Concatenate(axis=3)([unpool3, enc3_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        
        cat2 = tf.keras.layers.Concatenate(axis=3)([unpool2, enc2_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        

        unpool1 = self.unpool1(dec2_1)
        
        cat1 = tf.keras.layers.Concatenate(axis=3)([unpool1, enc1_2]) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.outlayer(dec1_1)
        x = self.output_resize_layer(x)
        return x
    
class Unet_v5(tf.keras.Model): 
    def __init__(self,base_map): 
        super(Unet_v5, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()
        self.mulp_l = tf.keras.layers.Multiply()
        self.base_map = base_map
        self.concat_n = 32

        self.input_resize_layer = tf.keras.layers.Resizing(
            256,
            256,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.output_resize_layer = tf.keras.layers.Resizing(
            82,
            67,
            interpolation='bilinear',
            crop_to_aspect_ratio=False,
        )

        self.mlp_stack = [
            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(self.concat_n),

            # 그냥 인풋데이터 처리하는 부분 # 이후 베이스맵이랑 베이스스모크 cnn레이어로 처리해서 아웃풋 채널수 맞춰주고 내적한 값으로 예측하도록
        ]

        self.left_stack = [
            tf.keras.layers.Conv2D(64, 3, padding='same',use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(64, 3, padding='same',use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(self.concat_n, 3, padding='same',use_bias=True),
        ]

        self.enc1_1 = CBR2d(out_channels = 64) 
        self.enc1_2 = CBR2d(out_channels=64)
            
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc2_1 = CBR2d(out_channels=128)
        self.enc2_2 = CBR2d(out_channels=128)
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        self.enc3_1 = CBR2d(out_channels=256)
        self.enc3_2 = CBR2d(out_channels=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )

        self.enc4_1 = CBR2d(out_channels=512)
        self.enc4_2 = CBR2d(out_channels=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), )
        
        
        self.enc5_1 = CBR2d(out_channels=1024)
        self.dec5_1 = CBR2d(out_channels=512)

        self.unpool4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=2,strides=(2, 2))

        self.dec4_2 = CBR2d(out_channels=512)
        self.dec4_1 = CBR2d(out_channels=256)

        self.unpool3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2,strides=(2, 2))

        self.dec3_2 = CBR2d(out_channels=256)
        self.dec3_1 = CBR2d(out_channels=128)

        self.unpool2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=2,strides=(2, 2))

        self.dec2_2 = CBR2d(out_channels=128)
        self.dec2_1 = CBR2d(out_channels=64)

        self.unpool1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2,strides=(2, 2))

        self.dec1_2 = CBR2d(out_channels=64)
        self.dec1_1 = CBR2d(out_channels=64)

        self.outlayer = tf.keras.layers.Conv2D(1,kernel_size = 1)
              
    def call(self, input): 

        # x = tf.multiply(tf.keras.layers.Reshape((1,1, 119))(input), self.base_map)
        x_left = tf.multiply(tf.ones([input.shape[0],1,1,1]), self.base_map)
        x_left = self.input_resize_layer(x_left)
        x = input

        # right_module
        for seq in self.mlp_stack:
            x = seq(x)

        for seq in self.left_stack:
            x_left = seq(x_left)
        

        x_after = tf.expand_dims(tf.einsum('bijk, bk ->bij', x_left, x),3)
        x = self.output_resize_layer(x_after)
        
        return x




class Lstm2dUnet(tf.keras.Model): 
    def __init__(self): 
        super(Lstm2dUnet, self).__init__()
        self.lose_mse = tf.keras.losses.MeanSquaredError()


        self.conv_lstm2d = tf.keras.layers.ConvLSTM2D(64, kernel_size=3, padding='same')

        self.down_stack = [
            Downsample(64, 3),  # (batch_size, 64, 64, 64)
            Downsample(128, 3),  # (batch_size, 32, 32, 128)
            Downsample(256, 3),  # (batch_size, 16, 16, 256)
            Downsample(512, 3),  # (batch_size, 8, 8, 512)
            Downsample(512, 3),  # (batch_size, 4, 4, 512)
        ]

        self.up_stack = [
            Upsample(512, 3, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            Upsample(512, 3),  # (batch_size, 8, 8, 1024)
            Upsample(256, 3),  # (batch_size, 16, 16, 512)
            Upsample(128, 3),  # (batch_size, 32, 32, 256)
            Upsample(64, 3),  # (batch_size, 64, 64, 128)
        ]

        self.initializer = tf.random_normal_initializer(0., 0.02)
        # self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,


        self.last_stack = [
            tf.keras.layers.Conv2DTranspose(64,4,
                                                strides=2,
                                                padding='same',
                                                # kernel_initializer=self.initializer,
                                                ),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(32, 3,
                                                # strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                ),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(1, 3,
                                                # strides=2,
                                                padding='same',
                                                # kernel_initializer=self.initializer,
                                                ),
                                                
                                                ]
        
    def call(self, input): 
        
        x = self.conv_lstm2d(input)
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            
            x = tf.keras.layers.Concatenate()([x, skip])
            
            

        for last_layer in self.last_stack:
            
            x = last_layer(x)
           
            
        
        return x
