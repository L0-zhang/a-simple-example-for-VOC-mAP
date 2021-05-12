# a-simple-example-for-VOC-mAP-calculation

Here provides a simple instance about  computing the mAP metric of VOC.

The dataset used could be found in https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release.

It has 2 classes,"hat" and "person".3 images are used from the dataset for an illustration.

And most of the code is based on https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/utils.py.

For more information,you could link to my blog https://blog.csdn.net/WANGWUSHAN/article/details/116709485.


The algorithm of coputing VOC mAP is Interpolating all points, as described in https://github.com/rafaelpadilla/Object-Detection-Metrics#11-point-interpolation.


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512192602909.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512192616889.png#pic_center)


### Through the YOLO_V3 trainned,the pred boxes are added to the images with the gt boxes respectively.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512154053372.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dBTkdXVVNIQU4=,size_16,color_FFFFFF,t_70#pic_center)


After executing the code,we get

- Table for class "hat" with 7 gt boxes：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512173423407.png#pic_center)


- Precision_Recall curve for class "hat"：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512190500167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dBTkdXVVNIQU4=,size_16,color_FFFFFF,t_70#pic_center)

- AP value for class "hat"：

AP=0.14285714∗1.0+0.14285715∗1.0+0.14285714∗1.0+0.14285714∗0.875+
0.14285714∗0.875+0.14285715∗0.875+0.14285714∗0.875
=0.92857142875


- Table for class "person" with 8 gt boxes：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512174009886.png#pic_center)

- Precision_Recall curve for class "person"：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512184304231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1dBTkdXVVNIQU4=,size_16,color_FFFFFF,t_70#pic_center)

- AP value for class "person"：

AP=0.125∗1.0+0.125∗1.0+0.125∗1.0+0.125∗1.0+
0.125∗1.0+0.125∗1.0+0.125∗1.0+0.125∗0
=0.875

### The mAP value is:

mAP=(0.92857+0.875)/2=0.901785



