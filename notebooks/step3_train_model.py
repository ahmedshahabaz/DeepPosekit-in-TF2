#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jgraving/deepposekit/blob/master/examples/step3_train_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # DeepPoseKit Step 3 - Train a model
# 
# This is step 3 of the example notebooks for using DeepPoseKit. This notebook shows you how to use your annotated data to train a deep learning model applying data augmentation and using callbacks for logging the training process and saving the best model during training.
# 
# **NOTE**: If you run into problems, you can help us improve DeepPoseKit by [opening an issue](https://github.com/jgraving/deepposekit/issues/new) or [submitting a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork)
# 
# **If you're using Colab**: make sure to go to the “Runtime” dropdown menu, select “Change runtime type” and select `GPU` in the "Hardware accelerator" drop-down menu

# If you haven't already installed DeepPoseKit you can run the next cell

# In[1]:


import tensorflow as tf
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia

from deepposekit.models import (StackedDenseNet,
                                DeepLabCut,
                                StackedHourglass,
                                LEAP)
from deepposekit.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint


import time
from os.path import expanduser

HOME = '/home/mithrandir/Downloads'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

'''
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()
'''


# Use the next cell to download the example data into your home directory

# There are a few example datasets to choose from:

# In[2]:


#glob.glob(HOME + '/deepposekit-data/**/**/*annotation*.h5')


# # Create a `DataGenerator`
# This creates a `DataGenerator` for loading annotated data. You can also look at the doc string for more explanation:
# 

# In[3]:


#DataGenerator?


# In[4]:


data_generator = DataGenerator(HOME + '/deepposekit-data/datasets/fly/annotation_data_release.h5')


# Indexing the generator, e.g. `data_generator[0]` returns an image-keypoints pair, which you can then visualize. 

# In[5]:


'''
image, keypoints = data_generator[0]

plt.figure(figsize=(5,5))
image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        plt.plot(
            [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
            [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
            'r-'
        )
plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)

plt.show()
'''


# # Create an augmentation pipeline
# DeepPoseKit works with augmenters from the [imgaug package](https://github.com/aleju/imgaug).
# This is a short example using spatial augmentations with axis flipping and affine transforms
# See https://github.com/aleju/imgaug for more documentation on augmenters.
# 
# `deepposekit.augment.FlipAxis` takes the `DataGenerator` as an argument to get the keypoint swapping information defined in the annotation set. When the images are mirrored keypoints for left and right sides are swapped to avoid "confusing" the model during training.

# In[6]:


augmenter = []

augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right 

sometimes = []
sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                            shear=(-8, 8),
                            order=ia.ALL,
                            cval=ia.ALL,
                            mode=ia.ALL)
                 )
sometimes.append(iaa.Affine(scale=(0.8, 1.2),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter.append(iaa.Sometimes(0.75, sometimes))
augmenter.append(iaa.Affine(rotate=(-180, 180),
                            mode=ia.ALL,
                            order=ia.ALL,
                            cval=ia.ALL)
                 )
augmenter = iaa.Sequential(augmenter)


# Load an image-keypoints pair, apply augmentation, visualize it. Rerun this cell to see multiple random augmentations.

# In[7]:


'''
image, keypoints = data_generator[0]
image, keypoints = augmenter(images=image, keypoints=keypoints)
plt.figure(figsize=(5,5))
image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
    if jdx > -1:
        plt.plot(
            [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
            [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
            'r-'
        )
plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)

plt.show()
'''


# # Create a `TrainingGenerator`
# This creates a `TrainingGenerator` from the `DataGenerator` for training the model with annotated data. The `TrainingGenerator` uses the `DataGenerator` to load image-keypoints pairs and then applies the augmentation and draws the confidence maps for training the model.
# 
# If you're using `StackedDenseNet`, `StackedHourglass`, or `DeepLabCut` you should set `downsample_factor=2` for 1/4x outputs or `downsample_factor=3` for 1/8x outputs (1/8x is faster). Here it is set to `downsample_factor=3` to maximize speed. If you are using `LEAP` you should set the `downsample_factor=0` for 1x outputs.
# 
# The `validation_split` argument defines how many training examples to use for validation during training. If your dataset is small (such as initial annotations for active learning), you can set this to `validation_split=0`, which will just use the training set for model fitting. However, when using callbacks, make sure to set `monitor="loss"` instead of `monitor="val_loss"`.
# 
# Visualizing the outputs in the next section also works best with `downsample_factor=0`.
# 
# You can also look at the doc string for more explanation:
# 

# In[8]:


#TrainingGenerator?


# In[9]:

'''
train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=0,
                                    augmenter=augmenter,
                                    sigma=5,
                                    validation_split=0.1, 
                                    use_graph=True,
                                    random_seed=1,
                                    graph_scale=1)
train_generator.get_config()
'''

# # Check the `TrainingGenerator` output
# This plots the training data output from the `TrainingGenerator` to ensure that the augmentation is working and the confidence maps look good. Rerun this cell to see random augmentations. 

# In[10]:


'''
n_keypoints = data_generator.keypoints_shape[0]
batch = train_generator(batch_size=1, validation=False)[0]
inputs = batch[0]
outputs = batch[1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
ax1.set_title('image')
ax1.imshow(inputs[0,...,0], cmap='gray', vmin=0, vmax=255)

ax2.set_title('posture graph')
ax2.imshow(outputs[0,...,n_keypoints:-1].max(-1))

ax3.set_title('keypoints confidence')
ax3.imshow(outputs[0,...,:n_keypoints].max(-1))

ax4.set_title('posture graph and keypoints confidence')
ax4.imshow(outputs[0,...,-1], vmin=0)
plt.show()

train_generator.on_epoch_end()
'''


# # Define a model
# Here you can define a model to train with your data. You can use our `StackedDenseNet` model, `StackedHourglass` model, `DeepLabCut` model, or the `LEAP` model. The default settings for each model should work well for most datasets, but you can customize the model architecture. The `DeepLabCut` model has multiple pretrained (on ImageNet) backbones available for using transfer learning, including the original ResNet50 (He et al. 2015)  as well as the faster MobileNetV2 (Sandler et al. 2018; see  also Mathis et al. 2019) and DenseNet121 (Huang et al. 2017). We'll select `StackedDenseNet` and set `n_stacks=2` for 2 hourglasses, with `growth_rate=32` (32 filters per convolution). Adjust the `growth_rate` and/or `n_stacks` to change model performance (and speed). You can also set `pretrained=True` to use transfer learning with `StackedDenseNet`, which uses a DenseNet121 pretrained on ImageNet to encode the images.

# In[11]:


from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP


# You can also look at the doc strings for any of the models to get more information:

# In[12]:


#StackedDenseNet?


# In[13]:


#DeepLabCut?


# In[14]:


train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=2,
                                    augmenter=augmenter,
                                    sigma=5,
                                    validation_split=0.1, 
                                    use_graph=True,
                                    random_seed=1,
                                    graph_scale=1)
train_generator.get_config()


# In[15]:


model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)
batch_size = 32


#model = DeepLabCut(train_generator, backbone="resnet50")
#model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
#model = DeepLabCut(train_generator, backbone="densenet121")

#model = LEAP(train_generator)
#model = StackedHourglass(train_generator)

#model.get_config()


# # Define callbacks to enhance model training
# Here you can define callbacks to pass to the model for use during training. You can use any callbacks available in `deepposekit.callbacks` or `tensorflow.keras.callbacks`
# 
# Remember, if you set `validation_split=0` for your `TrainingGenerator`, which will just use the training set for model fitting, make sure to set `monitor="loss"` instead of `monitor="val_loss"`.
# 
# 
# `Logger` evaluates the validation set (or training set if `validation_split=0` in the `TrainingGenerator`) at the end of each epoch and saves the evaluation data to a HDF5 log file (if `filepath` is set).

# In[17]:


logger = Logger(validation_batch_size=batch_size,
    # filepath saves the logger data to a .h5 file
    filepath=HOME + "/deepposekit-data/datasets/fly/log_densenet.h5"
)


# `ReduceLROnPlateau` automatically reduces the learning rate of the optimizer when the validation loss stops improving. This helps the model to reach a better optimum at the end of training.

# In[18]:


reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)


# `ModelCheckpoint` automatically saves the model when the validation loss improves at the end of each epoch. This allows you to automatically save the best performing model during training, without having to evaluate the performance manually.
# 

# In[19]:


model_checkpoint = ModelCheckpoint(
    HOME + "/deepposekit-data/datasets/fly/best_model_densenet.h5",
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)
#path = '/home/mithrandir/Downloads/deepposekit-data/datasets/fly/best_model_densenet.h5'
#model.save_weights(path)


# `EarlyStopping` automatically stops the training session when the validation loss stops improving for a set number of epochs, which is set with the `patience` argument. This allows you to save time when training your model if there's not more improvment.

# In[20]:


early_stop = EarlyStopping(
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    min_delta=0.001,
    patience=100,
    verbose=1,
    restore_best_weights=True
)


# Create a list of callbacks to pass to the model

# In[21]:

model.distribute_strategy = tf.distribute.MirroredStrategy()

#from tqdm.keras import TqdmCallback
#callbacks = [early_stop, reduce_lr, model_checkpoint, logger, TqdmCallback(verbose=1)]

callbacks = [early_stop, reduce_lr, model_checkpoint, logger]
#callbacks = [early_stop, reduce_lr, logger]


# # Fit the model
# 
# This fits the model for a set number of epochs with small batches of data. If you have a small dataset initially you can set `batch_size` to a small value and manually set `steps_per_epoch` to some large value, e.g. 500, to increase the number of batches per epoch, otherwise this is automatically determined by the size of the dataset.
# 
# The number of `epochs` is set to `epochs=200` for demonstration purposes. **Increase the number of epochs to train the model longer, for example `epochs=1000`**. The `EarlyStopping` callback will then automatically end training if there is no improvement. See the doc string for details:

# In[22]:


#model.fit?


# In[23]:



model.fit(
    batch_size=batch_size,
    validation_batch_size=batch_size,
    callbacks=callbacks,
    #epochs=1000, # Increase the number of epochs to train the model longer
    epochs=3,
    n_workers=8,
    steps_per_epoch=None,
)

#path = '/home/mithrandir/Downloads/deepposekit-data/datasets/fly/best_model_densenet.h5'
#model.save_weights(path)

# # Load the model and resume training
# 
# This loads the saved model and passes it the augmentation pipeline and `DataGenerator` from earlier.

# In[ ]:


model1 = load_model(
    HOME + "/deepposekit-data/datasets/fly/best_model_densenet.h5",
    augmenter=augmenter,
    generator=data_generator,
)

model1.distribute_strategy = tf.distribute.MirroredStrategy()

#model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)
#model.build(input_shape = (224, 224, None))
#model.build(input_shape=(16, None))



# To resume training, simply call `model.fit` again. We'll run it for another 30 `epochs`

# In[ ]:

#batch = next(train_generator)
#_ = model1.predict_generator(train_generator)

#_ = model1.predict(train_generator)
#model1.build(input_shape=(None, 192, 192, 1))
#model1(np.zeros((1,224,224,1)))

#model1.predict(np.zeros((1,192,192,1)))

#model1.train_model.call(np.zeros((1,192,192,1)))
#model1.train_model(np.zeros((1,192,192,1)))
#model1.call(np.zeros((1,192,192,1)))
#model1.load_weights(HOME + "/deepposekit-data/datasets/fly/best_model_densenet.h5")



model1.fit(
    batch_size=batch_size,
    validation_batch_size=batch_size,
    callbacks=callbacks,
    epochs=7,
    n_workers=8,
    steps_per_epoch=None,
)
