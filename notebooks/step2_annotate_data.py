#!/usr/bin/env python
# coding: utf-8

# # DeepPoseKit Step 2 - Annotate your data
# 
# This is step 2 of the example notebooks for using DeepPoseKit. This notebook shows you how to annotate your training data with user-defined keypoints using the saved data from step 1.

# If you haven't already installed DeepPoseKit you can run the next cell

# In[1]:


import sys
from deepposekit import Annotator
from os.path import expanduser
import glob
HOME = '/home/mithrandir/Downloads'


# Use the next cell to download the example data into your home directory

# Annotation Hotkeys
# ------------
# * <kbd>+</kbd><kbd>-</kbd> = rescale image by ±10%
# * <kbd>left mouse button</kbd> = move active keypoint to cursor location
# * <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> = move active keypoint 1px or 10px
# * <kbd>space</kbd> = change <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> mode (swaps between 1px or 10px movements)
# * <kbd>J</kbd><kbd>L</kbd> = next or previous image
# * <kbd><</kbd><kbd>></kbd> = jump 10 images forward or backward
# * <kbd>I</kbd>,<kbd>K</kbd> or <kbd>tab</kbd>, <kbd>shift</kbd>+<kbd>tab</kbd> = switch active keypoint
# * <kbd>R</kbd> = mark image as unannotated ("reset")
# * <kbd>F</kbd> = mark image as annotated ("finished")
# * <kbd>V</kbd> = mark active keypoint as visible
# * <kbd>esc</kbd> or <kbd>Q</kbd> = quit
# 
# # Annotate data
# Annotations are saved automatically. 
# The skeleton in each frame will turn blue when the frame is fully annotated. If there are no visible keypoints, this means the frame hasn't been annotated, so try clicking to position the keypoint in the frame.

# In[2]:


app = Annotator(datapath=HOME + '/deepposekit-data/datasets/fly/example_annotation_set.h5',
                dataset='images',
                skeleton=HOME + '/deepposekit-data/datasets/fly/skeleton.csv',
                shuffle_colors=False,
                text_scale=0.2)


# In[3]:


app.run()


# In[ ]:




