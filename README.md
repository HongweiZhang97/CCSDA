# Close-set Camera Style Distibution Align for Single Camera Person Re-identification

## Environment

The code has been tested on Pytorch 1.1.0 and Python 3.6. 

**Install other required packages**
```console
pip install -r requirements.txt
```
**Note:**
Our code is only tested with Python3.

We use ResNet-50 as the backbone. A pretrained model file is needed. Please put [this file](https://download.pytorch.org/models/resnet50-19c8e357.pth) in the reid/weights/pre_train directory. 

## Dataset Preparation 

**1. Download Market-SCT [BaiduYun](https://pan.baidu.com/s/1l3bPijoaMawA_7oaA3cuTQ) (password: 1234) and Duke-SCT [BaiduYun](https://pan.baidu.com/s/19xueemhc0YnQV8DCqFpWeg) (password: 1234)**

**2. Make new directories in data and organize them as follows:**
<pre>
+-- data
|   +-- market_sct
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
|   +-- duke_sct
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
</pre>

**3. Train with our Proposed CCSDA.**
## Generate style transfer images
Train the [CycleGAN-for-Camstyle](https://github.com/zhunzhong07/CamStyle/tree/master/CycleGAN-for-CamStyle) to generate style transfer images and then add them to the training set of the SCT datasets

## Train and test
To train with our proposed CCSDA, simply run train_bm.sh. 

To evaluate trained models, simply run test_bm.sh with single GPU.

**Note:**
We conducted all our experiments on single Tesla V100 GPU. Using multi GPU training models may cause performance degradation.
