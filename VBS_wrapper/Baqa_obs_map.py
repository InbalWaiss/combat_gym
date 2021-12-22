import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from os import path


srcImage = Image.open('Baqa.TIF')
img1 = np.array(srcImage.convert('L').resize((100, 100)))
img_org = np.array(srcImage.convert('L').resize((100, 100)))

for x in range(100):
    for y in range(100):
        if x+y>177 or x>88:
            img1[x,y] = img1[x,y]-8

img2 = cv2.bitwise_not(img1)
obsticals = cv2.inRange(img2, 163, 255)

thicken_obs_and_edges = cv2.bitwise_not(obsticals)
thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1

thicken = cv2.blur(thicken_obs_and_edges, ksize=(2,2))

thicken[0, :] = 1
thicken[99, :] = 1
thicken[:, 0] = 1
thicken[:, 99] = 1


fig, axs = plt.subplots(1,3)
axs[0].imshow(img_org)
axs[1].imshow(thicken_obs_and_edges)
axs[2].imshow(thicken)
plt.show()

#np.savetxt("BaqaObs.txt", thicken_obs_and_edges, fmt="%d")
np.savetxt("BaqaObs_thicken.txt", thicken, fmt="%d")