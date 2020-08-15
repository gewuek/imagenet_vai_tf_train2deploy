# imagenet_vai_tf_train2deploy<br />
This tutorial descibe about how to train a custom network from imagenet dataset and deploy it on ZCU102 via Vitis AI<br />

### Download and handle ImageNet data<br />
1. Download the imagenet training dataset and validation data set from http://academictorrents.com/collection/imagenet-2012 or http://www.image-net.org/download.php
2. Uncompress the two tar files:
```
tar -xvf ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_val.tar
```
3. It is a little complex to handle the training dataset, copy the ***x86/tar_image.py*** file into ***ILSVRC2012_img_train*** folder and run command:
```
python3 tar_image.py
```
4. 




















### Reference
https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf
