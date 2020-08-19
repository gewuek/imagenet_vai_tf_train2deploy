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

###  Train data<br />


### Quantize the model<br />

1. Freeze the model and change to pb file.<br />
```
python3 ./freeze_model.py
```
2. Edit the custom_network_input_fn.py to make sure you set the right address for ***calib_image_dir*** and ***calib_image_list***. Like the one on my test environment.<br />
```
calib_image_dir = "./ILSVRC2012_img_val/"
calib_image_list = "./calibration.txt"
```

3. Quantize the model.<br />
```
./decent_q.sh
```
4. Compile the quantized model to get the DPU ELF.<br />
```
./dnnc.sh
```
5. Check the generated ELF file from ***./resnet50/dpu_resnet50_0.elf***.<br />



















### Reference
https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf
