import numpy as np
import math
import cv2
import random
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from skimage import filters
from skimage import exposure

"""
Otsu thresholding
==================

This example illustrates automatic Otsu thresholding.
"""

img_filename = 'MAX_C2-ISP_FixTest_PFA_L-probe_40x_1-3NA_1-6x_20180608_1_MMStack_Pos0.ome.tif'

img = cv2.imread(img_filename)					
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)		
resized_img = cv2.resize(img, (300, 300))

val = filters.threshold_otsu(resized_img)
print(val)


hist, bins_center = exposure.histogram(resized_img)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(resized_img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(resized_img < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
