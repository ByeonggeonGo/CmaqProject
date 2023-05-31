import tensorflow as tf
from .customlayers import Upsample, Downsample
import time
from IPython import display
import matplotlib.pyplot as plt
import matplotlib

class Generator(tf.keras.Model): 
    def __init__(self,base_map): 
        super(Generator, self).__init__()

        

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
                                                kernel_initializer=self.initializer,
                                                ),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(32, 3,
                                                # strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                ),
            # tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(6, 3,
                                                # strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
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

        self.base_map = self.input_resize_layer(base_map)
        
    
    def call(self, input): 
        # x = self.input_resize_layer(input)
        x = input
        x_submap_layer_list = []
        for i in range(len(self.base_map[0,0,:])):
            x_submap_layer_list.append(tf.multiply(tf.keras.layers.Reshape((1,1, 100 + 119))(x), self.base_map[:,:,i:i+1]))
        x = tf.keras.layers.concatenate(x_submap_layer_list)

        # x = tf.multiply(tf.keras.layers.Reshape((1,1, 100 + 119))(x), self.base_map)  

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
    

class Discriminator(tf.keras.Model): 
    def __init__(self): 
        super(Discriminator, self).__init__()

        self.forward_stack = [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[82, 67, 6]),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1), # 바이너리 크로스엔트로피 구할 떄 로짓으로 구하므로 활성화함수x

        ]
    
    def call(self, input): 
        x = input
        for forward in self.forward_stack:
            x = forward(x)
        return x


class LstmGenerator(tf.keras.Model): 
    def __init__(self): 
        super(LstmGenerator, self).__init__()

        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_lstm2d = tf.keras.layers.ConvLSTM2D(32, kernel_size=3, padding='same', return_sequences=False, kernel_initializer=self.initializer)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()


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

        
        # self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,


        self.last_stack = [
            tf.keras.layers.Conv2DTranspose(64,4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                ),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(32, 3,
                                                # strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                ),
            # tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2D(1, 3,
                                                # strides=2,
                                                padding='same',
                                                kernel_initializer=self.initializer,
                                                ),
                                                
                                                ]
        

        
    
    def call(self, input): 
        # x = self.input_resize_layer(input)
        x = input
        x = self.conv_lstm2d(x)
  
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        # x = tf.multiply(tf.keras.layers.Reshape((1,1, 100 + 119))(x), self.base_map)  

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


class LstmDiscriminator(tf.keras.Model): 
    def __init__(self): 
        super(LstmDiscriminator, self).__init__()
        self.conv_lstm2d = tf.keras.layers.ConvLSTM2D(1, kernel_size=3, padding='same', return_sequences=False, )
  
        self.initializer = tf.random_normal_initializer(0., 0.02)

        self.forward_stack = [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[82, 67, 6]),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.ZeroPadding2D(),
            tf.keras.layers.Conv2D(512, 4, strides=1,
                                            kernel_initializer=self.initializer,
                                            use_bias=False),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.ZeroPadding2D(),

            tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=self.initializer)




        ]
    
    def call(self, input: list): 
        inp, tar = input[0], input[1]

        inp_1 = self.conv_lstm2d(inp)

       
        # tar = tf.keras.layers.Reshape([64,64,1])(tar)
        x = tf.keras.layers.concatenate([inp_1, tar])

        for forward in self.forward_stack:
            x = forward(x)
        return x

    

    









class GAN(tf.keras.Model): 
    def __init__(self, base_map, lambda_c): 
        super(GAN, self).__init__()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = Generator(base_map)
        self.discriminator = Discriminator()
        self.lambda_c = lambda_c

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output, gen_output, target):
        gan_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_c * l1_loss)

        return total_gen_loss
    
    def train(self, dataset, epochs, train_step, checkpoint, checkpoint_prefix, seed, batch_size):
        gen_l_list = []
        disc_l_list = []
        for epoch in range(epochs):
            start = time.time()

            for control_mt_batch, image_batch,_ in dataset:
                gen_l, disc_l = train_step(image_batch, control_mt_batch, batch_size)
                gen_l_list.append(gen_l)
                disc_l_list.append(disc_l)

            # GIF를 위한 이미지를 생성.,
            if (epoch + 1) % 10 == 0:
                display.clear_output(wait=True)
                self.generate_and_save_images(self.generator,
                                        epoch + 1,
                                        seed)

            #에포크가 지날 때마다 모델을 저장.
            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # 마지막 에포크가 끝난 후 생성
        
        display.clear_output(wait=True)
        self.generate_and_save_images(self.generator,
                                epochs,
                                seed)
        return gen_l_list, disc_l_list
    

    def generate_and_save_images(self, model, epoch, test_input):
  
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(7,7))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0][::-1], cmap = 'jet',norm = matplotlib.colors.Normalize())
            plt.axis('off')
        # os.path.join(proj_path,'models','training_checkpoints',"model_conmat_cmaq_1_checkpoint")
        # plt.savefig(os.path.join(proj_path,'plots','gen_plots','NOx_image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

        fig = plt.figure(figsize=(7,7))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 1][::-1], cmap = 'jet',norm = matplotlib.colors.Normalize())
            plt.axis('off')

        # plt.savefig(os.path.join(proj_path,'plots','gen_plots','SO2_image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

        fig = plt.figure(figsize=(7,7))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 2][::-1], cmap = 'jet',norm = matplotlib.colors.Normalize())
            plt.axis('off')

        # plt.savefig(os.path.join(proj_path,'plots','gen_plots','VOCs_image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

        fig = plt.figure(figsize=(7,7))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 3][::-1], cmap = 'jet',norm = matplotlib.colors.Normalize())
            plt.axis('off')

        # plt.savefig(os.path.join(proj_path,'plots','gen_plots','NH3_image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

    

class DCGAN(tf.keras.Model): 
    def __init__(self, lambda_c): 
        super(DCGAN, self).__init__()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = LstmGenerator()
        self.discriminator = LstmDiscriminator()
        self.lambda_c = lambda_c
        self.reshape_layer = tf.keras.layers.Reshape([64,64,1])

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output, gen_output, target):
        gan_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        tf.reduce_mean(tf.abs(target - gen_output))
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_c * l1_loss)

        return total_gen_loss
    
    def train(self, dataset, train_step,steps, checkpoint, checkpoint_prefix,):
        start = time.time()
        # 데이터셋 자체가 배치화되어있어서 여기서는 배치사이즈 고려하지 않음
        for step, (input_image, target) in dataset.repeat().take(steps).enumerate():
            target = self.reshape_layer(target)
            if (step) % 200 == 0:
                display.clear_output(wait=True)

                if step != 0:
                    print(f'Time taken for 200 steps: {time.time()-start:.2f} sec\n')

               
                print(f"Step: {step}")
            print(1)
            train_step(input_image, target, step)

            # Training step
            if (step+1) % 100 == 0:
                print('.', end='', flush=True)


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 1000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
        # 마지막 에포크가 끝난 후 생성
        
    def call(self, input): 
        x = self.generator(input)
        return x
    
class DCGAN_v2(tf.keras.Model): 
    def __init__(self, lambda_c, input_shape_cus, output_channels): 
        super(DCGAN_v2, self).__init__()

        self.input_shape_cus = input_shape_cus
        self.output_channels = output_channels

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lambda_c = lambda_c
        self.reshape_layer = tf.keras.layers.Reshape([64,64,1])

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output, gen_output, target):
        gan_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        tf.reduce_mean(tf.abs(target - gen_output))
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_c * l1_loss)

        return total_gen_loss
    
    def Generator(self,):

        inputs = tf.keras.layers.Input(shape=self.input_shape_cus)

        initializer_1 = tf.random_normal_initializer(0., 0.02)
        conv_lstm2d = tf.keras.layers.ConvLSTM2D(32, kernel_size=3, padding='same', return_sequences=False, kernel_initializer=initializer_1)
        batch_norm_1 = tf.keras.layers.BatchNormalization()
        relu_1 = tf.keras.layers.ReLU()

        down_stack = [
            downsample(64, 3),  # (batch_size, 64, 64, 64)
            downsample(128, 3),  # (batch_size, 32, 32, 128)
            downsample(256, 3),  # (batch_size, 16, 16, 256)
            downsample(512, 3),  # (batch_size, 8, 8, 512)
            downsample(512, 3),  # (batch_size, 4, 4, 512)
        ]

        up_stack = [
            upsample(512, 3, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 3),  # (batch_size, 8, 8, 1024)
            upsample(256, 3),  # (batch_size, 16, 16, 512)
            upsample(128, 3),  # (batch_size, 32, 32, 256)
            upsample(64, 3),  # (batch_size, 64, 64, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                )  # (batch_size, 256, 256, 3)  activation='tanh'

        x = inputs
        x = conv_lstm2d(x)
        
        x = batch_norm_1(x)
        x = relu_1(x)
        
        # x = data_augmentation(x)      

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def Discriminator(self):
        conv_lstm2d = tf.keras.layers.ConvLSTM2D(1, kernel_size=3, padding='same', return_sequences=False, )
        
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.input_shape_cus, name='input_image')
        tar = tf.keras.layers.Input(shape=[64, 64,1], name='target_image')

        inp_1 = conv_lstm2d(inp)

        print(inp_1)
        print(tar)

        x = tf.keras.layers.concatenate([inp_1, tar])  # (batch_size, 256, 256, channels*2)
        print(x)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down2)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
        print(last)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def train(self, dataset, train_step,steps, checkpoint, checkpoint_prefix,):
        start = time.time()
        plt.figure(figsize=(15, 5*int((steps/500))+1))
        j = 0
        # 데이터셋 자체가 배치화되어있어서 여기서는 배치사이즈 고려하지 않음
        for step, (input_image, target) in dataset.repeat().take(steps).enumerate():
            target = self.reshape_layer(target)
            if (step) % 200 == 0:

                if step != 0:
                    print(f'Time taken for 200 steps: {time.time()-start:.2f} sec\n')
                    start = time.time()
               
                print(f"Step: {step}")
            
            train_step(input_image, target, step)

            # Training step
            if (step+1) % 20 == 0:
                print('.', end='', flush=True)
                
                
            if (step+1) % 500 == 0:
                prediction = self.generator(input_image)
                display_list = [input_image[0][0][:,:,0][::-1], target[0][:,:,0][::-1], prediction[0][:,:,0][::-1]]
                title = [f'{step} step:Input Image_ first channel', f'{step} step:Ground Truth', f'{step} step:Predicted Image']
                
                for i in range(3):
                    plt.subplot(int((steps/500))+1, 3, j + i + 1)
                    plt.title(title[i])
                    # Getting the pixel values in the [0, 1] range to plot.
                    plt.imshow(display_list[i])
                    plt.axis('off')
                j += 3


            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 1000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            
        # 마지막 에포크가 끝난 후 생성


    def call(self, input): 
        x = self.generator(input)
        return x
    


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result