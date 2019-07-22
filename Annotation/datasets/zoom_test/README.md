This directory contains the results of running zoom.py on smfish.csv, which contains all the spot coordinates that were manually annotated over the smFISH widest image.

smfish.png is the parent image.

smfish_0.png, smfish_1.png, …, smfish_8.png are cropped in from smfish.png.

If the recursion continued to annother depth for smfish_1.png and smfish_7.png, we might get that smfish_1_0.png, smfish_1_1.png, and smfish_1_2.png are cropped in from smfish_1.png and smfish_7_0.png and smfish_7_1.png are cropped in from smfish_7.png.

For each crop image, a csv file of the same name gives the bounding box of the crop on the parent image.

Also for each crop image, the corresponding region on the parent image is blacked out. Therefore, all parent images have a corresponding [parent image name]_blacked.png file with their child crops blacked out. In this directory, the recursion only needed to go to one depth so there is only one parent image; the parent image with blacking out is smfish_blacked.png.