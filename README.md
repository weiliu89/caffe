# TextBoxes: A Fast Text Detector with a Single Deep Neural Network

### Introduction
This paper presents an end-to-end trainable fast scene text detector, named TextBoxes, which detects scene text with both high accuracy and efficiency in a single network forward pass, involving no post-process except for a standard nonmaximum suppression. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1611.06779).

### Citing TextBoxes
Please cite TextBoxes in your publications if it helps your research:

    @article{LiaoSBWL16TextBoxes,
      author    = {Minghui Liao and
                   Baoguang Shi and
                   Xiang Bai and
                   Xinggang Wang and
                   Wenyu Liu},
      title     = {TextBoxes: {A} Fast Text Detector with a Single Deep Neural Network},
      journal   = {CoRR},
      volume    = {abs/1611.06779},
      year      = {2016},
    }

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Test](#test)
4. [Train](#train)
5. [Performance](#performance)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/MhLiao/caffe.git
  
  cd caffe
  
  git checkout TextBoxes
  
  make -j8
  
  make py
  ```

### Download
1. Models trained on ICDAR 2013: [Dropbox link](https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1qY73XHq)
2. Fully convolutional reduced (atrous) VGGNet: [Dropbox link](https://www.dropbox.com/s/qxc64az0a21vodt/VGG_ILSVRC_16_layers_fc_reduced.caffemodel?dl=0) [BaiduYun link](http://pan.baidu.com/s/1slQyMiL)
3. Compiled mex file for non-maximum suppression(if you can not compile the 'polygon_intersect.cpp' correctly): [Dropbox link](https://www.dropbox.com/s/xtjuwvphxnz1nl8/polygon_intersect.mexa64?dl=0) [BaiduYun link](http://pan.baidu.com/s/1jIe9UWA)


### Test
1. Download the ICDAR 2013 DataSet
2. Download the Models trained on ICDAR 2013
3. Modify the related paths in the "examples/TextBoxes/test_icdar13.py"
4. run "python examples/test_icdar13.py"
5. To multi-scale test, you should use "test_icdar13_multi_scale.py" and "evaluation_nms.m"

### Train
1. Train about 50k iterions on Synthetic data which refered in the paper.
2. Train about 2k iterions on corresponding training data such as ICDAR 2013 and SVT.
3. For more information, such as learning rate setting, please refer to the paper.

### Performance
1. Using the given test code, you can achieve an F-measure of about 80% on ICDAR 2013 with a single scale.
2. Using the given multi-scale test code, you can achieve an F-measure of about 85% on ICDAR 2013 with a non-maximum suppression.
3. More performance information, please refer to the paper and Task1 and Task4 of Challenge2 on the ICDAR 2015 website: http://rrc.cvc.uab.es/?ch=2&com=evaluation

Please let me know if you encounter any issues.
