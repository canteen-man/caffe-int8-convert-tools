记录下我对fp32转int8或者fp16的理解，有不对地方欢迎指正。

tensort支持通过输出大量图像的办法或得int8或者fp16的权重。获得低bit权重能获得更快的计算速度。
最近在做FPGA，以我对PFGA的理解，xilinx的zynq系列soc中的FPGA内包含的DSP模块，这个DSP不是DSP芯片，是xilinx设计的DSP电路，可以完成乘法累加操作。

乘累加有个高大上的英文缩写，MACC。

xilinx的DSP可以在一个时钟周期内完成两次int8的乘累加，因为DSP输入位宽固定，输入数的位宽越小，一个时钟周期内算出来的结果也就越短，并行度也就越大。

再如以ARM为例，
这是我做的ARM neon指令集的3×3卷积程序，https://github.com/canteen-man/arm_neon_conv_3-3
int类型的整数最高可以以SIMD的计算方式达到8个寄存器同时计算，而float32则没有这么大的并行度。


所以如果cnn的权重都用int8，会根据不同的硬件设备加快计算速度。

但需要注意的是，这不能降低读写的速度，具体为什么，就要研究下float32是怎么转到int8的了。

参考了网上其他人的总结和分享，我大致理解float32转int8的流程如下:


首先对已有的float32做直方图统计，既然要做直方图统计就要分成每一个小柱形的统计区间，float32分布密集，就要自己设计统计小区间，按照nvidia的设计，是2048个bin。不过这2048个bin貌似是一侧的，以0位中心分为左右两侧。


不能直接把最大的两个值映射到int8的最大值127，因为饱和映射没有不饱和映射好，这个好理解，不能为了一个极端的值而压缩了另外许多值。

下一步，利用KL散度优化，开始不断的取阈值T，求出KL散度优化最小的T，即为要做的阈值。

最近发现channel pruning，BNN和int8量化的核心都是优化方法。

想想channel pruning，刚读论文的时候还很是惊讶，什么索套回归，之前都没听过。后来发现巧妙之处，太巧妙了，这索套回归的优化目的和cnn删除冗余卷积核后保证特征图重构误差最小的优化目的完全相同。


研究BNN时发现也是优化方法，研究转int8的时候又发现KL散度优化还是优化方法。

如果我要是博士的话，我就先把优化方法彻底的好好学一遍，然后没准哪个最优化的公式就能套进cnn，发个CCF A类。


得出阈值T后就能映射到int8了，所以就会有个scale。

但这个scale是从权重读进后再做scale，因为每一层的scale不同，每次做完卷积要再从int8回到float32，再从float32转到下一次int8，所以读取速度就没变。

按照ARM+FPGA的理解，就是AXI总线的传输效率没变，每次从DDR读图或者读权重的时间没变，读进BRAM后在折腾成int8，再输入DSP做计算，加速是加速在计算这步时间了。

貌似在哪听说tensorrt转换完的trt模型不能打印出权重，所以想拿tensort转换int8后用到其他地方的思路貌似是走不通了。



tensorrt转int8效果好的另一个原因是他是输入标定图像的，这个标定图像不是标定相机的那意思，就是把一堆图像输入，这样统计激活的分布就会更准，然后做KL散度优化的时候就更有代表性。








# Caffe-Int8-Convert-Tools

This convert tools is base on TensorRT 2.0 Int8 calibration tools,which use the KL algorithm to find the suitable threshold to quantize the activions from Float32 to Int8(-128 - 127).

We provide the Classification(SqueezeNet_v1.1) and Detection(MobileNet_v1 SSD 300) demos based on [ncnn](https://github.com/Tencent/ncnn)(It is a high-performance neural network inference framework optimized for the mobile platform),and the community ready to support this implment.

[ncnn-int8](https://github.com/Tencent/ncnn/pull/487)

## Reference

For details, please read the following PDF:

[8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) 

## HowTo

### Release version

```
$ python caffe-int8-convert-tool.py --help
usage: caffe-int8-convert-tool.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--gpu GPU]

find the pretrained caffe models int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained weights
  --mean MEAN           value of mean
  --norm NORM           value of normalize
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --gpu GPU             use gpu to forward
  
$ python caffe-int8-convert-tool.py --proto=squeezenet_v1.1.prototxt --model=squeezenet.caffemodel --mean 104 117 123 --images=ILSVRC2012_1k --output=squeezenet_v1.1.table --gpu=1
```

Pay attention to the type of images,it is just the original image format,such as jpg or jpeg,do not us the type of caffe dataset(lmdb).
**It is recommended to provide representative calibration dataset for the given model use case,such as the validation or test dataset.**

### How to use the output file(calibration.table)

For example in *squeezenet_v1_1.table*

```
conv1_param_0 138.066410
fire2/squeeze1x1_param_0 92.028103 // the conv layer's weight scale is 92.028
......
data 0.841264
conv1 0.295743
pool1 0.161700  // the pool layer's top blob scale is 0.1617
fire2/squeeze1x1 0.089383 // the conv layer's top blob scale is 0.0893
......
```

Three steps to implement the *fire2/squeeze1x1* layer int8 convolution:

1. Quantize the bottom_blob and weight:

   ```
   bottom_blob_int8 = bottom_blob_float32 * data_scale(0.1617)
   weight_int8 = weight_float32 * weight_scale(92.028)
   ```

2. Convolution_Int8:

   ```
   top_blob_int32 = bottom_blob_int8 * weight_int8
   ```

3. Dequantize the TopBlob_Int32 and add the bias:

   ```
   top_blob_float32 = top_blob_int32 / [data_scale(0.1617) * weight_scale(92.028)] + bias_float32
   ```

### Development version

The purpose of this tool(caffe-int8-convert-tool-dev.py) is to test new features,such as mulit-channels quantization depend on group num,sparse calculation and so on.

This format is already supported in the [ncnn](https://github.com/Tencent/ncnn) latest version.I will do my best to transform some common network models into [classification-dev](https://github.com/BUG1989/caffe-int8-convert-tools/tree/master/classification-dev)

```
python caffe-int8-convert-tool-dev.py -h
usage: caffe-int8-convert-tool.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--group GROUP] [--gpu GPU]

find the pretrained caffe models int8 quantize scale value

optional arguments:
  -h, --help            show this help message and exit
  --proto PROTO         path to deploy prototxt.
  --model MODEL         path to pretrained weights
  --mean MEAN           value of mean
  --norm NORM           value of normalize
  --images IMAGES       path to calibration images
  --output OUTPUT       path to output calibration table file
  --group GROUP         enable the group scale
  --gpu GPU             use gpu to forward
python caffe-int8-convert-tool-dev.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 103.94 116.78 123.68 --norm=0.017 --images=test/images/ --group=1
```

Although it's done,but the speed of group quanization is very slow......The difference from the release tool is that we try to get the int8_scale of bottom blob not the top blob. 

### How to use the output file(calibration-dev.table)

For example in *MobileNet_v1_dev.table*

```
conv1_param_0 156.639840
conv2_1/dw_param_0 0 72.129143 149.919382 // the convdw layer's weight scale every group is 0.0 72.129 149.919 ......
......
conv1 49.466518
conv2_1/dw 0 123.720796 48.705349 ...... // the convdw layer's bottom blob every group channel scale is 0.0 123.720 48.705 ......
......
```

## Accuracy and Performance

We used ImageNet2012 Dataset to complete some experiments.

| Type                | Detail                                                |
| ------------------- | ----------------------------------------------------- |
| Calibration Dataset | ILSVRC2012_img_test   1k                              |
| Test Dataset        | ILSVRC2012_img_val     5k                             |
| Framework           | ncnn-int8                                             |
| Support Layer       | Convolution3x3,Convolution1x1,ConvolutionDepthwise3x3 |

The following table show the Top1 and Top5 different between Float32 and Int8 inference.

|                 | FP32   |        | INT8      |           |
| --------------- | ------ | ------ | --------- | --------- |
| NETWORK         | Top1   | Top5   | Top1      | Top5      |
| SqueezeNet v1.1 | 57.86% | 79.86% | 57.36%    | 79.84%    |
| MobileNet v1    | 67.78% | 87.62% | 64.92%    | 85.22%    |
| MobileNet v2    | 70.20% | 89.20% | 69.00%    | 88.04%    |
| GoogleNet v1    | 67.70% | 88.32% | 67.64%    | 88.26%    |
| ResNet-18       | 65.50% | 86.46% | 65.48%    | 86.44%    |
| ResNet-50       | 71.68% | 89.94% | 71.38%    | 89.52%    |
| NETWORK         | Top1   | Top5   | Diff Top1 | Diff Top5 |
| SqueezeNet v1.1 | 57.86% | 79.86% | 0.50%     | 0.02%     |
| MobileNet v1    | 67.78% | 87.62% | 2.86%     | 2.40%     |
| MobileNet v2    | 70.20% | 89.20% | 1.06%     | 1.16%     |
| GoogleNet v1    | 67.70% | 88.32% | 0.06%     | 0.06%     |
| ResNet-18       | 65.50% | 86.46% | 0.02%     | 0.02%     |
| ResNet-50       | 71.68% | 89.94% | 0.30%     | 0.32%     |

The following table show the speedup between Float32 and Int8 inference.It should be noted that the winograd algorithm is not used in the Float32 inference.The Hardware Platform is Hisi3519(Cortex-A17@1.2GHz)

| Uint(ms) | SqueezeNet v1.1 | MobileNet v1 | MobileNet v2 | GoogleNet | ResNet18 | MobileNetv1 SSD |
| -------- | --------------- | ------------ | ------------ | --------- | -------- | --------------- |
| Float32  | 382             | 568          | 392          | 1662      | 1869     | 1120            |
| Int8     | 242             | 369          | 311          | 1159      | 1159     | 701             |
| Ratio    | x1.30           | x1.41        | x1.28        | x1.43     | x1.61    | x1.47           |

## Contributor

Thanks to our company [SenseNets](http://www.sensenets.com/home/) to support the open source project,and NVIDIA for providing the principle of correlation entropy,and ncnn's author [nihui](https://github.com/nihui) sharing his neural network inference framework.

Thanks to the help from the following friends:

Algorithm : [xupengfeixupf](https://github.com/xupengfeixupf), [JansonZhu](https://github.com/JansonZhu), wangxinwei, [lengmm](https://github.com/lengmm) 

Python : [daquexian](https://github.com/daquexian)

## License

BSD 3 Clause

