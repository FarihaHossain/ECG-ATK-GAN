from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from tensorflow import keras
from keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Add
from keras.layers i
mport Conv1DTranspose
from keras.layers import LeakyReLU, ReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout, GlobalMaxPool1D, MaxPooling1D
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

def define_discriminator(in_shape=(280,1), n_classes=4):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))    
    # downsample to 14x14
    fe = Conv1D(32, 3, strides=1, padding='same')(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(32, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    # normal
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(64, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 7x7
    fe = Conv1D(128, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    
    #downsample one more
    fe = Conv1D(256, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(256, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    
    # flatten feature maps
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # adversarial attack output
    out2 = Dense(2,  activation='softmax')(fe)
    # class label output
    out3 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2, out3],name="Discriminator")
    # compile model
    opt = Adam(lr=0.0001,beta_1=0.5)
    model.compile(loss=['mse','categorical_crossentropy','categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model

def novel_residual_block(X_input, filters,base):
    fe_add = Conv1D(filters, 3, strides=1, padding='same',name=base+'/conv')(X_input)
    fe = BatchNormalization(name=base+'/bn')(fe_add)
    fe = LeakyReLU(alpha=0.2,name=base+'/leakyReLU')(fe)
    fe = Add(name=base+'/add')([fe, fe_add])
    return fe

def skip_attention_block(X_input, filters,base):
    fe = Conv1D(filters, 2, strides=1, dilation_rate=2,padding='same',name=base+'/dil_conv')(X_input)
    fe = BatchNormalization(name=base+'/bn')(fe)
    fe = LeakyReLU(alpha=0.2,name=base+'/leakyReLU')(fe)
    return fe

def downsampling_block(X_input, filters,base):
    fe = Conv1D(filters, 3, strides=2, padding='same',name=base+'/conv')(X_input)
    fe = BatchNormalization(name=base+'/bn')(fe)
    fe = LeakyReLU(alpha=0.2,name=base+'/leakyReLU')(fe)
    return fe

def upsampling_block(X_input, filters,base):
    fe = Conv1DTranspose(filters, 3, strides=2, padding='same',name=base+'/conv')(X_input)
    fe = BatchNormalization(name=base+'/bn')(fe)
    fe = LeakyReLU(alpha=0.2,name=base+'/leakyReLU')(fe)
    return fe

# define the standalone generator model
def define_generator(latent_dim=(280,), signal_shape=(280,), label_shape=(4,)):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 32 #32
    ks = 3
    dropout = 0.25
    dim = 280 #
    # signal_input
    in_signal = Input(shape=signal_shape)
    si = Reshape((280,1))(in_signal)
    
    # label input
    in_label = Input(shape=label_shape)
    # embedding for categorical input
    li = Embedding(2, 50)(in_label)
    # linear multiplication
    n_nodes = 280 * 1
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((280,2))(li)
    
    
    # noise  input
    in_lat = Input(shape=latent_dim)
    lat = Reshape((1,280))(in_lat)
    # foundation for 7x7 image
    n_nodes = dim*depth
    gen = Dense(n_nodes)(lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, depth))(gen)
    # merge image gen and label input
    merge = Concatenate(name='merge')([gen, li,si]) #gen=280,32 x li=280,5 x si=280,1
    #merge = Concatenate()([gen, li]) #gen=280,32 li=280,5

    gen = novel_residual_block(merge, 32,'Block_1')
    #gen = Conv1D(32, 3, strides=1, padding='same',name='Block1/conv')(merge)
    #gen = BatchNormalization(name='Block1/bn')(gen)
    #gen = LeakyReLU(alpha=0.2,name='Block1/leakyReLU')(gen)
    skip_1 = skip_attention_block(gen,32,'Skip_1')
    gen = downsampling_block(gen, 32,'Down_1')
    # Residual + Downsampling
    gen = novel_residual_block(gen, 64,'Block_2')
    skip_2 = skip_attention_block(gen,64,'Skip_2')
    gen = downsampling_block(gen, 64,'Down_2')
    # Residual + Downsampling
    gen = novel_residual_block(gen, 128,'Block_3')
    skip_3 = skip_attention_block(gen,128,'Skip_3')
    gen = downsampling_block(gen, 128,'Down_3')
    # upsampling
    gen = upsampling_block(gen,128,'Up_1')
    gen = Add()([gen,skip_3])
    # upsampling
    gen = upsampling_block(gen,64,'Up_2')
    gen = Add()([gen,skip_2])
    # upsampling
    gen = upsampling_block(gen,32,'Up_3')
    gen = Add()([gen,skip_1])

    gen = Reshape((280,32))(gen)

    gen = Conv1D(1, 3, strides=1, name='Conv1D',padding='same')(gen)
    out_layer = Activation('sigmoid',name='Activation')(gen)

    model = Model([in_signal,in_lat, in_label], out_layer,name="Generator")
    #model = Model([in_lat, in_label], out_layer,name="Generator")
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model
    


def define_gan(g_model, d_model,latent_dim=(280,), signal_shape=(280,),label_shape=(4,)):
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    [out1,out2,out3] = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    #model = Model(g_model.input, gan_output)
    model = Model([g_model.input[0],g_model.input[1],g_model.input[2]],[out1,out2,out3])
    #model = Model([g_model.input[0],g_model.input[1]],[out1,out2])
    #model = Model([in_signal,in_lat, in_label],[out1,out2])
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss=['mse','categorical_crossentropy', 'categorical_crossentropy'], optimizer=opt,loss_weights=[1,10,10])
    model.summary()
    return model    