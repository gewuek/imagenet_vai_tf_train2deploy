# imagenet_vai_tf_train2deploy<br />
This tutorial descibe about how to train a custom network from imagenet dataset and deploy it on ZCU102 via Vitis AI<br />
It is full imagenet_2012 dataset so some of the operations would take long time.

### Download and handle ImageNet data<br />
1. Git clone this repository to your local machine with Vitis AI 1.2 installed, I would suggest you to do that inside the Vitis-AI folder so that the ***imagenet_vai_tf_train2deploy*** folder would be inside the docker workspace. e.g. Mine is ***/home/wuxian/wu_software/Vitis-AI/wu_project/imagenet_vai_tf_train2deploy/***.<br />
2. Download the imagenet training dataset and validation data set from http://academictorrents.com/collection/imagenet-2012 or http://www.image-net.org/download.php. Move these 2 files to the ***imagenet_vai_tf_train2deploy/x86/*** folder. <br />
3. Uncompress the two tar files:<br />
```
mkdir ILSVRC2012_img_train
mkdir ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_train.tar -C ./ILSVRC2012_img_train
tar -xvf ILSVRC2012_img_val.tar -C ./ILSVRC2012_img_val
```
4. It is a little complex to handle the training dataset, copy the ***x86/tar_image.py*** file into ***imagenet_vai_tf_train2deploy/x86/ILSVRC2012_img_train/*** folder and run the script inside the ***ILSVRC2012_img_train*** folder:<br />
```
python3 tar_image.py
rm *.tar
```
5. Go back to ***imagenet_vai_tf_train2deploy/x86*** folder, edit the ***load_data.py*** and make sure the ***LABEL_FOLDER_CSV*** and ***TRAINING_DATA_DIR*** are configured properly:<br />
```
LABEL_FOLDER_CSV = './label_wordnetid.csv'
TRAINING_DATA_DIR = './ILSVRC2012_img_train'
```
6. Run the load_data.py to create the file to store the image paths.<br />
```
python3 ./load_data.py
```
***You should get a training_image_path_label.txt file to store the image paths and labels***<br />

###  Train data<br />

1. Go to your ***Vitis-AI*** repository and launch the docker with following command:<br />
```
./docker_run.sh xilinx/vitis-ai-gpu:latest
```
2. Select the ***vitis-ai-tensorflow*** environment.<br />
```
conda activate vitis-ai-tensorflow
```
***Or you can just install the GPU version of TensorFlow 1.15 at host following this [TensorFlow GPU install tutorial](https://www.tensorflow.org/install/gpu)***<br />
3. Go to your ***/workspace/.../imagenet_vai_tf_train2deploy/x86*** floder, and edit the ***train_data.py*** file. Make sure the ***VAL_IMAGE_DIR*** is set properly.<br />
```
VAL_IMAGE_DIR = './ILSVRC2012_img_val/'
```
4. You can also modify the training parameters if necessary:<br />
```
BATCH_SIZE = 128
ds = image_label_ds.shuffle(buffer_size=5000)
```
BATCH_SIZE and buffer_size can be reduced if you don't have a GPU with big GPU memory.<br />
You can change ***weights*** from:<br />
```
model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True, input_shape= (224,224,3), classes=1000)
```
to
```
model = keras.applications.resnet50.ResNet50(weights=None, include_top=True, input_shape= (224,224,3), classes=1000)
```
if you would like to train from scratch not a pre-trained weights.<br />
***Note: If use a pre-trained weights it takes about 3 round(epoch) to acheive a 60%+ accuracy. If you train from scratch you may need more than 20 epoches to get a good accuracy. And if you train from scratch you can modify the _read_img_function function to use a more simple pre-processing. In the meanwhile you should also modify the calibration code and deploy code accordingly. For now the _read_img_function just uses the same pre-processing operations as the pre-trained model.***<br />

5. Run the train script:<br />
```
python3 ./train_data.py
```
6. After finishing the training, find the model file ***custom_network.h5*** inside ***imagenet_vai_tf_train2deploy/x86*** folder.<br />


### Quantize and compile the model<br />

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
5. Copy the generated ELF file from ***imagenet_vai_tf_train2deploy/x86/resnet50/dpu_resnet50_0.elf*** to ***imagenet_vai_tf_train2deploy/arm/tf_resnet50/model_for_zcu102/*** folder.<br />

### Deploy model on ZCU102 board<br />

1. Download the [Xilinx released VAI 1.2 ZCU102 Image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.2.0.img.gz) and burn on a SD card then boot up ZCU102 board with this SD card.<br />
2. Download the [vitis-ai_v1.2_dnndk_sample_img.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.2_dnndk_sample_img.tar.gz) and transfer the package to ZCU102 using SCP command.<br />
3. Uncompress the folder:<br />
```
tar -xzvf vitis-ai_v1.2_dnndk_sample_img.tar.gz
```
4. Go to the uncompressed folder ***vitis_ai_dnndk_samples*** and transfer ***imagenet_vai_tf_train2deploy/arm/tf_resnet50/*** folder from host to ***vitis_ai_dnndk_samples** folder on board.
5. Go to ***tf_reset50*** and run the following commmand to build the application:
```
./build.sh zcu102
```
6. Connect board with SSH and enable X11 forward. Run following command to check the result:
```
./tf_resnet50
```
The running result would like below:<br />
![renet50_test_result.PNG](/pic_for_readme/renet50_test_result.PNG)<br />

***I don't take much time on the training and comparing the difference between different APIs from training to quantization and deployment. So the result are not in a high accuracy. And you could use other ConvNet to replace the ResNet50 and modify the pre-process functions if necessary***



















### Reference<br />
https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf<br />
https://github.com/Xilinx/Vitis-AI<br />
https://github.com/gewuek/flower_classification_vai_tf_dataset<br />
https://github.com/sony/nnabla-examples/tree/master/imagenet-classification<br />
https://github.com/keras-team/keras-applications/tree/1.0.8/keras_applications<br />
https://github.com/gewuek/flower_classification_vai_tf_dataset


