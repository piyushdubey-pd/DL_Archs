from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras.layers.merge import concatenate

def naive_module(first_layer,f1,f2,f3):
  conv1=Conv2D(f1,(1,1), padding='same', activation='relu')(first_layer)
  conv3=Conv2D(f3,(3,3), padding='same', activation='relu')(first_layer)
  conv5=Conv2D(f3,(5,5), padding='same', activation='relu')(first_layer)
  pool=MaxPooling2D((3,3), strides=(1,1), padding='same')(first_layer)
  return concatenate([conv1,conv3,conv5,pool],axis=-1)
insep=Input(shape=(64,64,3))
layer=naive_module(insep,###)
model=Model(inputs=insep, outputs=layer)
model.summary()
