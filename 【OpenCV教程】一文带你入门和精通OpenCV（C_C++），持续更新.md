
@[TOC](文章目录（OpenCV版本4.6.0）)

---
# 1.数据类型

CV_8U:占8位的unsigned

CV_8UC(n):占8位的unsigned char

CV_8UC1:占8位的unsigned char 一通道

CV_8UC2:占8位的unsigned char 二通道

CV_8UC3:占8位的unsigned char 三通道

CV_8UC4:占8位的unsigned char 四通道

CV_8S:占8位的signed

CV_8SC(n):占8位的signed char

CV_8SC1:占8位的signed char 一通道

CV_8SC2:占8位的signed char 二通道

CV_8SC3:占8位的signed char 三通道

CV_8SC4:占8位的signed char 四通道

CV_16U:占16位的unsigned

CV_16UC(n):占16位的unsigned char

CV_16UC1:占16位的unsigned char 一通道

CV_16U2:占16位的unsigned char 二通道

CV_16U3:占16位的unsigned char 三通道

CV_16U4:占16位的unsigned char 四通道

CV_16S:占16位的signed

CV_16SC(n):占16位的signed char

CV_16SC1:占16位的signed char 一通道

CV_16SC2:占16位的signed char 二通道

CV_16SC3:占16位的signed char 三通道

CV_16SC4:占16位的signed char 四通道

CV_16F:占16位的float

CV_16FC(n):占16位的float char

CV_16FC1:占16位的float char 一通道

CV_16FC2:占16位的float char 二通道

CV_16FC3:占16位的float char 三通道

CV_16FC4:占16位的float char 四通道

CV_32S:占32位的signed

CV_32SC(n):占32位的signed char

CV_32SC1:占32位的signed char 一通道

CV_32SC2:占32位的signed char 二通道

CV_32SC3:占32位的signed char 三通道

CV_32SC4:占32位的signed char 四通道

CV_32F:占32位的float

CV_32FC(n):占32位的float char

CV_32FC1:占32位的float char 一通道

CV_32FC2:占32位的float char 二通道

CV_32FC3:占32位的float char 三通道

CV_32FC4:占23位的float char 四通道

CV_64F:占64位的float

CV_64FC(n):占64位的float char

CV_64FC1:占64位的float char 一通道

CV_64FC2:占64位的float char 二通道

CV_64FC3:占64位的float char 三通道

CV_64FC4:占64位的float char 四通道

![](https://img-blog.csdnimg.cn/img_convert/0db745ea6e64294aefefebc92c0147ca.png)

# 2.矩阵基本操作

## 2.1 全零矩阵

```cpp
CV_NODISCARD_STD static MatExpr Mat::zeros(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr Mat::zeros(Size size, int type);

CV_NODISCARD_STD static MatExpr Mat::zeros(int ndims, const int* sz, int type);
//not recommended
```

rows:行数

cols:列数

type:数据类型（CV_16F）

size:Size（宽（列数），高（行数））

* Size与Mat中的成员函数.size()的返回值，有相同的数据类型，是[宽*高]。
* Mat中的成员变量.size，与以上二者不同，是 rows*cols

## 2.2 全一矩阵

```cpp
CV_NODISCARD_STD static MatExpr Mat::ones(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr Mat::ones(Size size, int type);

CV_NODISCARD_STD static MatExpr Mat::ones(int ndims, const int* sz, int type);
//not recommended
```

rows:行数

cols:列数

type:数据类型（CV_16F）

size:Size（宽（列数），高（行数））

## 2.3 单位矩阵

```cpp
CV_NODISCARD_STD static MatExpr Mat::eye(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr Mat::eye(Size size, int type);
```

rows:行数

cols:列数

type:数据类型（CV_16F）

size:Size（宽（列数），高（行数））

## 2.4 矩阵转置

```cpp
MatExpr Mat::t() const;
```

## 2.5 求逆矩阵

```cpp
MatExpr Mat::inv(int method=DECOMP_LU) const;
```

## 2.6 逗号式分隔创建矩阵，常用于自定义卷积核

```cpp
template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols)
    : Mat(_rows, _cols, traits::Type<_Tp>::value)
{
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols, const _Tp& value)
    : Mat(_rows, _cols, traits::Type<_Tp>::value)
{
    *this = value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz)
    : Mat(_sz.height, _sz.width, traits::Type<_Tp>::value)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz, const _Tp& value)
    : Mat(_sz.height, _sz.width, traits::Type<_Tp>::value)
{
    *this = value;
}
```

以下为使用实例

```
Mat a=Mat_<int>(2,2)<<(1,2,3,4);
Mat b=Mat_<double>(Size(2,2))<<(1,2,3,4);
```

**注意** ：给出的数据类型必须是基本数据类型，如int,double。不能是CV_16F等。

## 2.7 矩阵定义(只列出常用的)

```cpp
Mat::Mat() CV_NOEXCEPT;
Mat::Mat(int rows, int cols, int type);
Mat::Mat(Size size, int type);
Mat::Mat(int rows, int cols, int type, const Scalar& s);
Mat::Mat(Size size, int type, const Scalar& s);
Mat::Mat(const std::vector<int>& sizes, int type);
Mat::Mat(const std::vector<int>& sizes, int type, const Scalar& s);
Mat::Mat(const Mat& m);

void Mat::create(int rows, int cols, int type);
void Mat::create(Size size, int type);
void Mat::create(const std::vector<int>& sizes, int type);
```

rows:行数

cols:列数

type:数据类型（CV_16F）

size:Size（宽（列数），高（行数））

### 2.7.1 数据类型Scalar

* Scalar(gray)
* Scalar(blue,green,red)

## 2.8 通过ptr与at函数遍历矩阵

### 2.8.1 Vec类型

```cpp
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;
 
typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;
 
typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;
 
typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;
 
typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;
 
typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
```

以下为实例

```cpp
Mat a(Size(2560,1440),CV_8UC3);
for(int i=0;i<a.rows;i++){
      for(int j=0;j<a.cols;j++){
          a.ptr(i,j)[0]=0;
          a.ptr(i,j)[1]=0;
          a.ptr(i,j)[2]=255;
      }
}
for(int i=0;i<a.rows;i++){
      for(int j=0;j<a.cols;j++){
          a.ptr<Vec3b>(i,j)[0]=0;
          a.ptr<Vec3b>(i,j)[1]=0;
          a.ptr<Vec3b>(i,j)[2]=255;
      }
}
for(int i=0;i<a.rows;i++){
      for(int j=0;j<a.cols;j++){
          a.at<Vec3b>(i,j)[0]=0;
          a.at<Vec3b>(i,j)[1]=0;
          a.at<Vec3b>(i,j)[2]=255;
      }
}
```

* 用ptr访问可以不加Vec类型，**ptr访问是最快的**
* 用at访问必须加Vec类型，**at访问比ptr略微慢一些**

## 2.9 通过迭代器遍历矩阵(easy but very very slow)

```cpp
Mat a(Size(2560,1440),CV_8UC3);
for(auto iter=a.begin<Vec3b>();iter!=a.end<Vec3b>();iter++){
      iter[0]=255;
      iter[1]=0;
      iter[2]=0;
}
```

# 3.图像基本操作

## 3.1 图片读取

```cpp
CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );
enum ImreadModes {

       IMREAD_UNCHANGED            = -1, 
//!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.

       IMREAD_GRAYSCALE            = 0,  
//!< If set, always convert image to the single channel grayscale image (codec internal conversion).

       IMREAD_COLOR                = 1,  
//!< If set, always convert image to the 3 channel BGR color image.

       IMREAD_ANYDEPTH             = 2,  
//!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.

       IMREAD_ANYCOLOR             = 4,  
//!< If set, the image is read in any possible color format.

       IMREAD_LOAD_GDAL            = 8,  
//!< If set, use the gdal driver for loading the image.

       IMREAD_REDUCED_GRAYSCALE_2  = 16, 
//!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.

       IMREAD_REDUCED_COLOR_2      = 17, 
//!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.

       IMREAD_REDUCED_GRAYSCALE_4  = 32, 
//!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.

       IMREAD_REDUCED_COLOR_4      = 33, 
//!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.

       IMREAD_REDUCED_GRAYSCALE_8  = 64, 
//!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.

       IMREAD_REDUCED_COLOR_8      = 65, 
//!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.

       IMREAD_IGNORE_ORIENTATION   = 128 
//!< If set, do not rotate the image according to EXIF's orientation flag.
     };
```

## 3.2 创建窗口

```cpp
CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
```

winname(window name)：窗体名

## 3.3 图片显示

```cpp
CV_EXPORTS_W void imshow(const String& winname, InputArray mat);
```

winname(window name)：窗体名

若窗体未创建，会自动进行创建

```cpp
CV_EXPORTS_W int waitKey(int delay = 0);
```

控制图片的展示时间，如设置delay=0，则表示一直展示，按SPACE停止展示

如设置delay不为0，则表示停留delay毫秒

## 3.4 图片保存

```cpp
CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());
```

filename：保存的文件名

## 3.5 视频输入输出

```cpp
CV_WRAP explicit VideoCapture::VideoCapture(const String& filename, int apiPreference = CAP_ANY);
  
CV_WRAP explicit VideoCapture::VideoCapture(const String& filename, int apiPreference, const std::vector<int>& params);

CV_WRAP explicit VideoCapture::VideoCapture(int index, int apiPreference = CAP_ANY);

CV_WRAP explicit VideoCapture::VideoCapture(int index, int apiPreference, const std::vector<int>& params);

CV_WRAP VideoWriter::VideoWriter(const String& filename, int fourcc, double fps,Size frameSize, bool isColor = true);

CV_WRAP VideoWriter::VideoWriter(const String& filename, int fourcc, double fps, const Size& frameSize,const std::vector<int>& params);

CV_WRAP VideoWriter::VideoWriter(const String& filename, int apiPreference, int fourcc, double fps,const Size& frameSize, const std::vector<int>& params);

//fps:帧率
//frameSize：输出视频中每一帧的尺寸
```

### 3.5.1 filename

影片档案名称（例如video.avi）

图片序列（例如img_%02d.jpg，将读取像这样的样本img_00.jpg, img_01.jpg, img_02.jpg, …）

视频流的网址（例如protocol://host:port/script_name?script_params|auth）。请注意，每个视频流或IP摄像机源均具有其自己的URL方案。请参考源流的文档以了解正确的URL。

### 3.5.2 index

**要打开的视频捕获设备的ID**。要使用默认后端打开默认摄像头，只需传递0。

当apiPreference为CAP_ANY时，使用camera_id + domain_offset（CAP_ *）向后兼容有效。

###  3.5.3 fourcc
用于编码视频文件的编码器，通过VideoWriter::fourcc函数获得

```cpp
    CV_WRAP static int fourcc(char c1, char c2, char c3, char c4);
```
|代码 |含义  |
|--|--|
|  VideoWriter::fourcc('P','I','M','1')  | MPEG-1编码，输出文件拓展名avi |
|  VideoWriter::fourcc('X','V','I','D')  | MPEG-4编码，输出文件拓展名avi           |
|  VideoWriter::fourcc('M','P','4','V')  | 旧MPEG-4编码，输出文件拓展名avi          |
|  VideoWriter::fourcc('I','4','2','0')  | YUV编码，输出文件拓展名avi           |
|  VideoWriter::fourcc('X','2','6','4')  | MPEG-4编码，输出文件拓展名mp4           |
|  VideoWriter::fourcc('T','H','E','O')  | ogg vorbis编码，输出文件拓展名ogv          |
|  VideoWriter::fourcc('F',L','V','1')  | flash video编码，输出文件拓展名flv          |

### 3.5.4 apiPreference（not  important）

首选使用的Capture API后端。如果有多个可用的读取器实现，则可以用于实施特定的读取器实现。

设置读取的摄像头编号，默认CAP_ANY=0,自动检测摄像头。多个摄像头时，使用索引0,1,2，…进行编号调用摄像头。 apiPreference = -1时单独出现窗口，选取相应编号摄像头。

### 3.5.5 演示

```cpp
VideoCapture video("demo.mp4");
Mat fps;
video.read(fps);
VideoWriter video_out("demo_out.avi",VideoWriter::fourcc('P','I','M','1'),30,fps.size());
    while (1){
        Mat fps;
        video>>fps;
        //video.read(fps);
        fps>>video_out;
        //video_out.write(fps);
        imshow("video",fps);
        waitKey(10);//控制帧率
    }
```

## 3.6 通道分离与合并

### 3.6.1 分离

#### API（一）

```cpp
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);
```

src(source)：输入图像。

mvbegin(mat vector begin)：分离后的Mat数组。

#### API（二）

```cpp
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);
```

m(mat)：输入图像。

mv(mat vector)：分离后的的Mat数组，**可以使用STL容器vector。**

### 3.6.2 合并

#### API（一）

```cpp
CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);
```

mv(mat vector)：欲合并的图像数组。

count：欲合并的图像的个数

dst(destination)：输出图片。

#### API（二）

```cpp
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);
```

mv(mat vector)：欲合并的图像数组，**可以使用STL容器vector。**

dst(destination)：输出图片。

## 3.7 图片色彩模式转换
###  3.7.1 API
```
CV_EXPORTS_W void cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 );
```

code：转换码

### 3.7.2 转换类型和转换码

-  RGB和BGR（opencv默认的彩色图像的颜色空间是BGR）颜色空间的转换

cv::COLOR_BGR2RGB

cv::COLOR_RGB2BGR

cv::COLOR_RGBA2BGRA

cv::COLOR_BGRA2RGBA

-   向RGB和BGR图像中增添alpha通道

cv::COLOR_RGB2RGBA

cv::COLOR_BGR2BGRA

-   从RGB和BGR图像中去除alpha通道

cv::COLOR_RGBA2RGB

cv::COLOR_BGRA2BGR

-  从RBG和BGR颜色空间转换到灰度空间

cv::COLOR_RGB2GRAY

cv::COLOR_BGR2GRAY

cv::COLOR_RGBA2GRAY

cv::COLOR_BGRA2GRAY

-  从灰度空间转换到RGB和BGR颜色空间

cv::COLOR_GRAY2RGB

cv::COLOR_GRAY2BGR

cv::COLOR_GRAY2RGBA

cv::COLOR_GRAY2BGRA

-  RGB和BGR颜色空间与BGR565颜色空间之间的转换

cv::COLOR_RGB2BGR565

cv::COLOR_BGR2BGR565

cv::COLOR_BGR5652RGB

cv::COLOR_BGR5652BGR

cv::COLOR_RGBA2BGR565

cv::COLOR_BGRA2BGR565

cv::COLOR_BGR5652RGBA

cv::COLOR_BGR5652BGRA

-  灰度空间与BGR565之间的转换

cv::COLOR_GRAY2BGR555

cv::COLOR_BGR5552GRAY

-  RGB和BGR颜色空间与CIE XYZ之间的转换

cv::COLOR_RGB2XYZ

cv::COLOR_BGR2XYZ

cv::COLOR_XYZ2RGB

cv::COLOR_XYZ2BGR

-  RGB和BGR颜色空间与uma色度（YCrCb空间）之间的转换

cv::COLOR_RGB2YCrCb

cv::COLOR_BGR2YCrCb

cv::COLOR_YCrCb2RGB

cv::COLOR_YCrCb2BGR

-  RGB和BGR颜色空间与HSV颜色空间之间的相互转换

cv::COLOR_RGB2HSV

cv::COLOR_BGR2HSV

cv::COLOR_HSV2RGB

cv::COLOR_HSV2BGR

-  RGB和BGR颜色空间与HLS颜色空间之间的相互转换

cv::COLOR_RGB2HLS

cv::COLOR_BGR2HLS

cv::COLOR_HLS2RGB

cv::COLOR_HLS2BGR

-   RGB和BGR颜色空间与CIE Lab颜色空间之间的相互转换

cv::COLOR_RGB2Lab

cv::COLOR_BGR2Lab

cv::COLOR_Lab2RGB

cv::COLOR_Lab2BGR

-  RGB和BGR颜色空间与CIE Luv颜色空间之间的相互转换

cv::COLOR_RGB2Luv

cv::COLOR_BGR2Luv

cv::COLOR_Luv2RGB

cv::COLOR_Luv2BGR

-   Bayer格式（raw data）向RGB或BGR颜色空间的转换

cv::COLOR_BayerBG2RGB

cv::COLOR_BayerGB2RGB

cv::COLOR_BayerRG2RGB

cv::COLOR_BayerGR2RGB

cv::COLOR_BayerBG2BGR

cv::COLOR_BayerGB2BGR

cv::COLOR_BayerRG2BGR

cv::COLOR_BayerGR2BGR

## 3.8 改变图片的对比度和亮度

### 3.8.1 概述

```cpp
Mat.ptr(i,j)=Mat.ptr(i,j)*a+b
```

a：控制对比度增益

b：控制亮度增益

### 3.8.2 手动（使用saturate_cast函数确保输出值不溢出范围）

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);
for(int i=0;i<xuenai.rows;i++){
        for(int j=0;j<xuenai.cols;j++){
            for(int k=0;k<xuenai.channels();k++) {
                xuenai.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(xuenai.at<Vec3b>(i, j)[k] *                 1.2 + 30);
            }
        }
    }
imshow("xuenai_convertTo",xuenai);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/9b2596c3d34b202dce725c0c5a9dba15.png)

### 3.8.3 调用API：Mat::convertTo

```cpp
void Mat::convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;
```

```
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);
xuenai.convertTo(xuenai,-1,1.2,30);
imshow("xuenai_convertTo",xuenai);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/7745ecb5a494f7742ce8ca6d1acda14f.png)

可以看到效果是一样的

## 3.9 图片混合

```cpp
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);
```

src(source1)：输入图片1

alpha：src1的权重

src2(source2)：输入图片2

beta：src2的权重

gamma：额外的增量

dst(destination)：输出图片

dtype(destination type)：输出图片的数据类型，-1表示与输入图片一致

## 3.10 图片尺寸调整

```cpp
CV_EXPORTS_W void resize( InputArray src, OutputArray dst,
                          Size dsize, double fx = 0, double fy = 0,
                          int interpolation = INTER_LINEAR );
```

src(source)：输入图片

dst(destination)：输出图片

dsize(destination size)：输出图片的尺寸

fx：x方向(width方向)的缩放比例，如果它是0，那么它就会按照(double)dsize.width/src.cols来计算；

fy：y方向(height方向)的缩放比例，如果它是0，那么它就会按照(double)dsize.height/src.rows来计算；

interpolation：插值算法的选择

### 3.10.1 插值算法(not important)

```cpp
enum InterpolationFlags{
    /** nearest neighbor interpolation */
    INTER_NEAREST        = 0,
    /** bilinear interpolation */
    INTER_LINEAR         = 1,
    /** bicubic interpolation */
    INTER_CUBIC          = 2,
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    INTER_AREA           = 3,
    /** Lanczos interpolation over 8x8 neighborhood */
    INTER_LANCZOS4       = 4,
    /** Bit exact bilinear interpolation */
    INTER_LINEAR_EXACT = 5,
    /** Bit exact nearest neighbor interpolation. This will produce same results as
    the nearest neighbor method in PIL, scikit-image or Matlab. */
    INTER_NEAREST_EXACT  = 6,
    /** mask for interpolation codes */
    INTER_MAX            = 7,
    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    source image, they are set to zero */
    WARP_FILL_OUTLIERS   = 8,
    /** flag, inverse transformation

    For example, #linearPolar or #logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    WARP_INVERSE_MAP     = 16
};
```

### 3.10.2 注意事项

使用注意事项：

* dsize和fx/fy不能同时为0

1. 指定dsize的值，让fx和fy空置直接使用默认值。
2. 让dsize为0，指定好fx和fy的值，比如fx=fy=0.5，那么就相当于把原图两个方向缩小一倍。

## 3.11 图像金字塔（常用于神经网络的池化层，对图像进行成倍的放大或缩小）

```cpp
CV_EXPORTS_W void pyrDown( InputArray src, OutputArray dst,
                           const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );
//缩小一倍
```

src(source)：输入图片

dst(destination)：输出图片

dstsize(destination size)：输出图片的尺寸，默认自动调整

borderType：边界填充方式，默认为黑边。如果没有设置dstsize，则不会出现黑边，因为已经进行了自动调整

```cpp
CV_EXPORTS_W void pyrUp( InputArray src, OutputArray dst,
                         const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );
//放大一倍
```

src(source)：输入图片

dst(destination)：输出图片

dstsize(destination size)：输出图片的尺寸，默认自动调整

borderType：边界填充方式，默认为黑边。如果没有设置dstsize，则不会出现黑边，因为已经进行了自动调整

## 3.12 二值化（对灰度图）

```cpp
CV_EXPORTS_W double threshold( InputArray src, OutputArray dst,
                               double thresh, double maxval, int type );
```

src(source)：输入图片

dst(destination)：输出图片

thresh(threshold)：阈值

maxval(max value)：最大值

type：阈值类型

### 3.12.1 阈值类型

```cpp
enum ThresholdTypes {
    THRESH_BINARY     = 0, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_BINARY_INV = 1, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f]
    THRESH_TRUNC      = 2, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_TOZERO     = 3, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_TOZERO_INV = 4, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_MASK       = 7,
    THRESH_OTSU       = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
    THRESH_TRIANGLE   = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value
};
```

#### 阈值二值化（Threshold Binary）

　　首先指定像素的灰度值的阈值，遍历图像中像素值，如果像素的灰度值大于这个阈值，则将这个像素设置为最大像素值(8位灰度值最大为255)；若像素的灰度值小于阈值，则将该像素点像素值赋值为0。公式以及示意图如下：

![](https://img-blog.csdnimg.cn/img_convert/ff06947279d2a3b26264114f05528468.webp?x-oss-process=image/format,png)

#### 阈值反二值化（Threshold Binary Inverted）

　　首先也要指定一个阈值，不同的是在对图像进行阈值化操作时与阈值二值化相反，当像素的灰度值超过这个阈值的时候为该像素点赋值为0；当该像素的灰度值低于该阈值时赋值为最大值。公式及示意图如下：

![](https://img-blog.csdnimg.cn/img_convert/879cf4f4a68b0014379ca013021cf1f8.webp?x-oss-process=image/format,png)

####  截断（Truncate）

　　给定像素值阈值，在图像中像素的灰度值大于该阈值的像素点被设置为该阈值，而小于该阈值的像素值保持不变。公式以及示意图如下：

![](https://img-blog.csdnimg.cn/img_convert/4c4f0ea5608db2397e01041fea4a59e2.webp?x-oss-process=image/format,png)

####  阈值取零（Threshold To Zero）

　　与截断阈值化相反，像素点的灰度值如果大于该阈值则像素值不变，如果像素点的灰度值小于该阈值，则该像素值设置为0.公式以及示意图如下：

![](https://img-blog.csdnimg.cn/img_convert/52764c3ebcb5320522b068d775362963.webp?x-oss-process=image/format,png)

####  阈值反取零（Threshold To Zero Inverted）

　　像素值大于阈值的像素赋值为0，而小于该阈值的像素值则保持不变，公式以及示意图如下：

![](https://img-blog.csdnimg.cn/img_convert/c171b53f81490cd480590e6f500d38cb.webp?x-oss-process=image/format,png)

## 3.13 图片裁剪

### 3.13.1 方式一

```cpp
inline
Mat Mat::operator()( const Rect& roi ) const
{
    return Mat(*this, roi);
}
```

以下为实例

```cpp
Mat xuenai = imread("xuenai.jpg");
resize(xuenai,xuenai,Size(1000,1000));
imshow("xuenai", xuenai);
Mat tuanzi(xuenai,(Rect(0,0,500,1000)));
imshow("tuanzi",tuanzi);
waitKey();
```

### 3.13.2 方式二

```cpp
Mat::Mat(const Mat& m, const Rect& roi);
```

以下为实例

```cpp
Mat xuenai = imread("xuenai.jpg");
resize(xuenai,xuenai,Size(1000,1000));
imshow("xuenai", xuenai);
Mat tuanzi(xuenai(Rect(0,0,500,1000)));
imshow("tuanzi",tuanzi);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/02eccb3bc7e3a06bbdce62fc0735efea.png)

### 3.13.3 Rect类构造

```cpp
template<typename _Tp> inline
Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    : x(_x), y(_y), width(_width), height(_height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2)
{
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}
```

## 3.14 基本变换

### 3.14.1 翻转

```cpp
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);
```

src(source)：输入图片

dst(destination)：输出图片

flipCode：翻转类型

```
flipcode==0;//上下翻转
flipcod>0;//左右翻转
flipcode<0;//上下加左右翻转,等价于旋转180°
```

#### 效果

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);
Mat xuenai_flip(xuenai.size(), xuenai.type());
flip(xuenai, xuenai_flip, 0);
imshow("xuenai_flip", xuenai_flip);
waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/8fac1ce122c3ba3dc7ba08dfac14fc05.png)

### 3.14.2 90°旋转

```cpp
CV_EXPORTS_W void rotate(InputArray src, OutputArray dst, int rotateCode);
enum RotateFlags {
    ROTATE_90_CLOCKWISE = 0, //!<Rotate 90 degrees clockwise
    ROTATE_180 = 1, //!<Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //!<Rotate 270 degrees clockwise
};
```

src(source)：输入图片

dst(destination)：输出图片

rotateCode：旋转类型

#### 效果

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);
Mat xuenai_rotate(xuenai.size(), xuenai.type());
rotate(xuenai, xuenai_rotate, ROTATE_180);
imshow("xuenai_rotate", xuenai_rotate);
waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/46c763929af7e67f3c6909042466340f.png)

## 3.15 仿射变换与透射变换

```cpp
CV_EXPORTS_W void warpAffine( InputArray src, OutputArray dst,
                              InputArray M, Size dsize,
                              int flags = INTER_LINEAR,
                              int borderMode = BORDER_CONSTANT,
                              const Scalar& borderValue = Scalar());
```

src(source)：输入图片

dst(destination)：输出图片

M：变换矩阵

dsize(destination size)：输出图片的尺寸，**若不对输出图片的尺寸进行调整，那么很可能会出现黑边**

flags：插值算法

borderMode：边界外推法

borderValue：填充边界的值

```cpp
CV_EXPORTS_W void warpPerspective( InputArray src, OutputArray dst,
                                   InputArray M, Size dsize,
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const Scalar& borderValue = Scalar());
```
src(source)：输入图片

dst(destination)：输出图片

M：变换矩阵

dsize(destination size)：输出图片的尺寸，**若不对输出图片的尺寸进行调整，那么很可能会出现黑边**

flags：插值算法

borderMode：边界外推法

borderValue：填充边界的值

### 3.15.1 平移

只需将变换矩阵M设置成如下形式：

```cpp
float delta_x=200,delta_y=200;
float  M_values[]={1,0,delta_x,
                   0,1,delta_y};
Mat M(Size(3,2),CV_32F,M_values);
```

delta_x：x方向上的偏移量

delta_y：y方向上的偏移量

M_values：**必须是浮点类型的数组对象**

M：**必须是CV_32F，不能用逗号式分隔创建**

#### 效果

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai",xuenai);
double  M_values[]={1,0,200,
                    0,1,200};
Mat M(Size(3,2), CV_64F,M_values);
Mat xuenai_shift(xuenai.size(),xuenai.type());
warpAffine(xuenai,xuenai_shift,M,xuenai.size());
imshow("xuenai_shift",xuenai_shift);
waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/2ec0dea0c7b256210c37921ae457aaf5.png)

### 3.15.2 任意角度旋转

#### 获得变换矩阵M

```cpp
inline
Mat getRotationMatrix2D(Point2f center, double angle, double scale)
{
    return Mat(getRotationMatrix2D_(center, angle, scale), true);
}
```

center：旋转中心点的坐标

angle：逆时针偏角

scale：生成图与原图之比

#### 效果

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);
Mat M= getRotationMatrix2D(Point2f(xuenai.cols/2,xuenai.rows/2),45,1);
Mat xuenai_rotate(xuenai.size(),xuenai.type());
warpAffine(xuenai,xuenai_rotate,M,xuenai.size());
imshow("xuenai_flip",xuenai_rotate);
```

![](https://img-blog.csdnimg.cn/img_convert/6ad57811155de9587dbaab1fcc91cff3.png)

### 3.15.3 仿射（不破坏几何关系）

#### 获得变换矩阵M

```cpp
CV_EXPORTS Mat getAffineTransform( const Point2f src[], const Point2f dst[] );
```

src[](source[])：输入图片的**坐标点集，含三个坐标点**

dst[](destination[])：三个坐标点变换的**目标位置**

**三个点要一一对应**

### 3.15.4 透射（破坏几何关系）

#### 已知变换后图片，逆推变换矩阵M

```cpp
CV_EXPORTS_W Mat getPerspectiveTransform(InputArray src, InputArray dst, int solveMethod = DECOMP_LU);
```

src(source)：输入图片

dst(destination)：输出图片

#### 获得变换矩阵M

```cpp
CV_EXPORTS Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[], int solveMethod = DECOMP_LU);
```

src[](source[])：输入图片的**坐标点集，含四个坐标点**

dst[](destination[])：四个坐标点变换的**目标位置**

**四个点要一一对应**

####  效果

```cpp
        Mat origin = imread("origin.jpg");
        Point2f point2F_origin[4]={Point2f (405,105),Point2f(2469,217),Point2f(2573,3489),Point2f(349,3547)};
        Point2f point2F_tansform[4]={Point2f (0,0),Point2f(2500,0),Point2f(2500,3500),Point2f(0,3500)};
        Mat M=getPerspectiveTransform(point2F_origin,point2F_tansform);
        Mat transfrom(origin.size(),origin.type());
        warpPerspective(origin,transfrom,M,Size(2500,3500));
        resize(origin,origin,Size(500,700));
        resize(transfrom,transfrom,Size(500,700));
        imshow("origin",origin);
        imshow("transform",transfrom);
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/974f93dbdde4431a8218245816741fd3.png#pic_center)


# 4.滤波及边缘检测

## 4.1 均值滤波

### 4.1.1 卷积核形状

```cpp
Mat kernal=Mat::ones(Size(ksize,ksize),CV_64F)/(ksize*ksize);
```

### 4.1.2 API

```cpp
CV_EXPORTS_W void blur( InputArray src, OutputArray dst,
                        Size ksize, Point anchor = Point(-1,-1),
                        int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

ksize(kernal size)：卷积核大小，**必须是正奇数**

anchor：滤波器中心像素位置，取（-1，-1）表示几何中心

borderType：边界填充方式，默认为黑边

### 4.1.3 效果

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai",xuenai);
Mat xuenai_blur(xuenai.size(),xuenai.type());
blur(xuenai,xuenai_blur,Size(3,5));
imshow("xuenai_blur",xuenai_blur);
waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/3e3a93194ade7a2ec7ba56ae29323c5e.png)

## 4.2 高斯滤波

### 4.2.1 卷积核形状

二维高斯函数表述为：

![](https://img-blog.csdnimg.cn/img_convert/a2c8dd843c2c2253e6ab5d81aa44b464.jpeg)

对应图形：

![](https://img-blog.csdnimg.cn/img_convert/89a95474bda24917c6c4e4eedebbbbcd.png)

代码实现（不区分sigmaX与sigmaY）

```cpp
void GetGaussianKernel(Mat kernal, const int ksize,const double sigma)  
{  
    const double PI=4.0*atan(1.0); //圆周率π赋值  
    int center= ksize/2;  
    double sum=0;  
    for(int i=0;i<ksize;i++)  
    {  
        for(int j=0;j<ksize;j++)  
        {  
            kernal.ptr(i,j)=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));  
            sum+=kernal.ptr(i,j);  
        }  
    }  

    for(int i=0;i<ksize;i++)  
    {  
        for(int j=0;j<ksize;j++)  
        {  
            kernal.ptr(i,j)/=sum;  
        }  
    }  
    return ;  
}
```

### 4.2.2 API

```cpp
CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,
                                double sigmaX, double sigmaY = 0,
                                int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

ksize(kernal size)：卷积核大小。**如果这个值我们设其为非正数，那么OpenCV会从第五个参数sigmaSpace来计算出它来。**

sigmaX：x方向上的标准差

sigmaY：y方向上的标准差。**默认输入量为0，则将其设置为等于sigmaX，如果两个轴的标准差均为0，则根据输入的高斯滤波器尺寸计算标准偏差。**

borderType：边界填充方式，默认为黑边

### 4.2.3 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);
    Mat xuenai_Gauss(xuenai.size(),xuenai.type());
    GaussianBlur(xuenai,xuenai_Gauss,Size(-1,-1),10);
    imshow("xuenai_Gauss",xuenai_Gauss);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/3961d2c582edfe68df4e98e4a743c219.png)

## 4.3 中值滤波

### 4.3.1 原理

取滤波器内的中值作为输出，**可以很好的抑制椒盐噪声**

### 4.3.2 API

```cpp
CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );
```

src(source)：输入图片

dst(destination)：输出图片

ksize(kernal size)：卷积核边长，**必须是正奇数**

### 4.3.3 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);
    Mat xuenai_median(xuenai.size(),xuenai.type());
    medianBlur(xuenai,xuenai_median,5);
    imshow("xuenai_median",xuenai_median);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/c3eb7d88bd4bb0e44aa794369f50c541.png)

## 4.4 高斯双边滤波

### 4.4.1 原理

双边滤波器的好处是可以做边缘保存（edge preserving），一般用高斯滤波去降噪，会较明显地模糊边缘，对于高频细节的保护效果并不明显。双边滤波器顾名思义比高斯滤波多了一个高斯方差sigma－d，它是基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波。

![](https://img-blog.csdnimg.cn/img_convert/33513cc34283f3251b2a4d420c900b31.png)

### 4.4.2 API

```cpp
CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,
                                   double sigmaColor, double sigmaSpace,
                                   int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

d：卷积核边长。**如果这个值我们设其为非正数，那么OpenCV会从第五个参数sigmaSpace来计算出它来。**

sigmaColor：颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。

sigmaSpace：坐标空间中滤波器的sigma值，坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，卷积核大小已被指定且与sigmaSpace无关。否则，d正比于sigmaSpace。

borderType：边界填充方式，默认为黑边

### 4.4.3 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);
    Mat xuenai_bilateral(xuenai.size(),xuenai.type());
    bilateralFilter(xuenai,xuenai_bilateral,-1,100,10);
    imshow("xuenai_bilateral",xuenai_bilateral);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/8a771a085b9410de2ba8186bb8c0ee8a.png)

## 4.5 获取用来形态学操作的滤波器

```cpp
CV_EXPORTS_W Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));
enum MorphShapes {
    MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
    MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
                       //!< \f[E_{ij} = \begin{cases} 1 & \texttt{if } {i=\texttt{anchor.y } {or } {j=\texttt{anchor.x}}} \\0 & \texttt{otherwise} \end{cases}\f]
    MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
                      //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
};
```

shape：滤波器形状

ksize(kernal size)：滤波器大小

anchor：滤波器中心像素位置，取（-1，-1）表示几何中心

## 4.6 腐蚀和膨胀（对二值图）

### 4.6.1 原理

腐蚀：取滤波器内的最小值作为输出

膨胀：取滤波器内的最大值作为输出

### 4.6.2 腐蚀API

```cpp
CV_EXPORTS_W void erode( InputArray src, OutputArray dst, InputArray kernel,
                         Point anchor = Point(-1,-1), int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const Scalar& borderValue = morphologyDefaultBorderValue() );
```

src(source)：输入图片

dst(destination)：输出图片

kernal：滤波器矩阵

anchor：滤波器中心像素位置，取（-1，-1）表示几何中心

iterations：执行erode函数的次数，默认执行一次

borderType：边界填充方式

borderValue：填充边界的值

### 4.6.3 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_erode(xuenai.size(),xuenai.type());
    erode(xuenai_threshold,xuenai_erode,kernal);
    imshow("xuenai_erode",xuenai_erode);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/f4cee99e5ac7db4894d4f95f6a8ca857.png)

### 4.6.4 膨胀API

```cpp
CV_EXPORTS_W void dilate( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );
```

src(source)：输入图片

dst(destination)：输出图片

kernal：滤波器矩阵

anchor：滤波器中心像素位置，取（-1，-1）表示几何中心

iterations：执行erode函数的次数

borderType：边界填充方式

borderValue：填充边界的值

### 4.6.5 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_dilate(xuenai.size(),xuenai.type());
    dilate(xuenai_threshold,xuenai_dilate,kernal);
    imshow("xuenai_dilate",xuenai_dilate);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/caf9a26636f9740e51f17509b5fef700.png)

## 4.7 形态学操作（对二值图）

### 4.7.1 API

```cpp
CV_EXPORTS_W void morphologyEx( InputArray src, OutputArray dst,
                                int op, InputArray kernel,
                                Point anchor = Point(-1,-1), int iterations = 1,
                                int borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );
```

src(source)：输入图片

dst(destination)：输出图片

op：变换类型

kernal：滤波器矩阵

anchor：滤波器中心像素位置，取（-1，-1）表示几何中心

iterations：执行erode函数的次数

borderType：边界填充方式

borderValue：填充边界的值

### 4.7.2 变换类型

```cpp
enum MorphTypes{
    MORPH_ERODE    = 0, //腐蚀

    MORPH_DILATE   = 1, //膨胀

    MORPH_OPEN     = 2, //开
                        
    MORPH_CLOSE    = 3, //闭
                        
    MORPH_GRADIENT = 4, //形态学梯度
                        
    MORPH_TOPHAT   = 5, //顶帽
                        
    MORPH_BLACKHAT = 6, //黑帽

    MORPH_HITMISS  = 7  //击中击不中变换

};
```

### 4.7.3 开

#### 原理

对输入图片先进行腐蚀，然后进行膨胀。可以用来**屏蔽与滤波器大小相当的亮部**。

#### 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_morphology(xuenai.size(),xuenai.type());
    morphologyEx(xuenai_threshold,xuenai_morphology,MORPH_OPEN,kernal);
    imshow("xuenai_morphology",xuenai_morphology);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/fd0e88c5798ffe5ea8fdec2c3b679661.png)

### 4.7.4 闭

#### 原理

对输入图片先进行膨胀，然后进行腐蚀。可以用来**屏蔽与滤波器大小相当的暗部**。

#### 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));
    
    Mat xuenai_morphology(xuenai.size(),xuenai.type());
    morphologyEx(xuenai_threshold,xuenai_morphology,MORPH_CLOSE,kernal);
    imshow("xuenai_morphology",xuenai_morphology);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/a7c3a6e91c41e87a41b8b08aae5e1db3.png)

### 4.7.5 顶帽

#### 原理

对输入图片先进行开操作，然后原图-开操作图。可以用来**提取与滤波器大小相当的亮部**。

#### 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_morphology(xuenai.size(),xuenai.type());
    morphologyEx(xuenai_threshold,xuenai_morphology,MORPH_TOPHAT,kernal);
    imshow("xuenai_morphology",xuenai_morphology);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/62e00915fed78453ff24da0217ab2341.png)

### 4.7.6 黑帽

#### 原理

对输入图片先进行闭操作，然后闭操作图-原图。可以用来**提取与滤波器大小相当的暗部**。

#### 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_morphology(xuenai.size(),xuenai.type());
    morphologyEx(xuenai_threshold,xuenai_morphology,MORPH_BLACKHAT,kernal);
    imshow("xuenai_morphology",xuenai_morphology);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/ada2d8f8881f2f98318328370604a267.png)

### 4.7.7 形态学梯度

#### 原理

膨胀图与腐蚀图之差。可以用来 **提取边界轮廓** ，但提取效果比不上专业的边缘检测算法。

#### 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");

    Mat xuenai_gray(xuenai.size(),xuenai.type());
    cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);
    Mat xuenai_threshold(xuenai.size(),xuenai.type());
    threshold(xuenai_gray,xuenai_threshold,100,255,THRESH_BINARY);
    imshow("xuenai_threshold",xuenai_threshold);

    Mat kernal=getStructuringElement(MORPH_RECT,Size(3,3));

    Mat xuenai_morphology(xuenai.size(),xuenai.type());
    morphologyEx(xuenai_threshold,xuenai_morphology,MORPH_GRADIENT,kernal);
    imshow("xuenai_morphology",xuenai_morphology);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/6af0c45fc131cafaa30323e4af6dd70f.png)

### 4.7.8 击中击不中变换

#### 原理

击中击不中变换由下面三步构成：

用结构元素B1来腐蚀输入图像

用结构元素B2来腐蚀输入图像的补集

前两步结果的与运算

结构元素B1和B2可以结合为一个元素B。例如：

![](https://img-blog.csdnimg.cn/img_convert/e06a9e05f206e3912353642fd254008c.png)

结构元素：左B1（击中元素），中B2（击不中元素），右B（两者结合）

本例中，我们寻找这样一种结构模式，中间像素属于背景，其上下左右属于前景，其余领域像素忽略不计（背景为黑色，前景为白色）。然后用上面的核在输入图像中找这种结构。从下面的输出图像中可以看到，输入图像中只有一个位置满足要求。

![](https://img-blog.csdnimg.cn/img_convert/8478a5898a1cef4166176c7b7597839a.png)

输入二值图像

![](https://img-blog.csdnimg.cn/img_convert/0ddc8fc8f461e91d6fb59f67b13745bd.png)

输出二值图像

## 4.8 边缘检测：选择合适的输出深度

参照以下表格

| int sdepth    | int ddepth           |
| :-------------- | :--------------------- |
| CV_8U         | CV_16S/CV_32F/CV_64F |
| CV_16U/CV_16S | CV_32F/CV_64F        |
| CV_32F        | CV_32F/CV_64F        |
| CV_64F        | CV_64F               |
### 4.8.1 normalize归一化函数

```cpp
CV_EXPORTS_W void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());
```

src(source)：输入数组

dst(destination)：输出数组

alpha ：**如果norm_type为NORM_MINMAX ，则alpha为最小值或最大值；如果norm_type为其他类型，则为归一化要乘的系数**

beta ：**如果norm_type为NORM_MINMAX ，则beta为最小值或最大值；如果norm_type为其他类型，beta被忽略.**

norm_type ：归一化类型

dtype ：输出数组的深度，若输入-1则表示与src一致。**如果不能判断需要的深度，则可以输入-1然后使用convertScaleAbs绝对值化，这也是最推荐的做法，而不推荐自己判断深度**。

mask ：掩码，用于指示函数是否仅仅对指定的元素进行操作。大小必须与src保持一致。具体用法见8.1.4
####  归一化类型（只介绍常用的四种）

```cpp
enum NormTypes {
                NORM_INF       = 1,
                NORM_L1        = 2,
                NORM_L2        = 4,
                NORM_L2SQR     = 5,
                NORM_HAMMING   = 6,
                NORM_HAMMING2  = 7,
                NORM_TYPE_MASK = 7, //!< bit-mask which can be used to separate norm type from norm flags
                NORM_RELATIVE  = 8, //!< flag
                NORM_MINMAX    = 32 //!< flag
};
 ```

 - NORM_L1        
![在这里插入图片描述](https://img-blog.csdnimg.cn/377e1301b4e04957bf51a4a6f984dd99.png#pic_center)
- NORM_L2        
![在这里插入图片描述](https://img-blog.csdnimg.cn/43eb9e08647946eea3008c772bb7787f.png#pic_center)

- NORM_INF       

![在这里插入图片描述](https://img-blog.csdnimg.cn/6b7aa2691e9740fd982f79d38cfc7025.png#pic_center)

- NORM_MINMAX(recommended)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c14ae1eed1e949b0852b93585f71a792.png#pic_center)

### 4.8.2 convertScaleAbs绝对值化

```cpp
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);
```

src(source)：输入图片

dst(destination)：输出图片

## 4.9 sobel（对灰度图）

### 4.9.1 卷积核形状（ksize=3）

```cpp
Mat kernalX=Mat_<int>(Size(3,3))<<(-1,0,1
                                    -2,0,2
                                    -1,0,1);
Mat kernalY=Mat_<int>(Size(3,3))<<(-1,-2,1
                                     0,0,0
                                     1,2,1);
```

### 4.9.2 API

```cpp
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize = 3,
                         double scale = 1, double delta = 0,
                         int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

ddepth(destination depth)：输出图片的深度（CV_16F）

dx：x方向导数的阶数，一般取1

dy：y方向导数的阶数，一般取1

ksize：卷积核边长，默认为3

scale：生成图与原图的缩放比例，默认为1

delta：额外的增量，默认为0

borderType：边界填充方式，默认为黑边

### 4.9.3 流程

1. 用cvtColor函数转**灰度图**
2. **在x,y方向上分别各调用一次Sobel**
3. **用convertScaleAbs函数转换到CV_8U，否则无法显示**
4. **用addWeighted函数把两张输出图片加在一起**

### 4.9.4 同时在x,y方向上调用Sobel和分开调用的效果对比

```cpp
Mat xuenai = imread("xuenai.jpg");
imshow("xuenai", xuenai);

//转灰度图
Mat xuenai_gray(xuenai.size(),xuenai.type());
cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);

//同时在x,y方向上调用Sobel
Mat xuenai_sobel1(xuenai.size(),xuenai.type());
Sobel(xuenai_gray,xuenai_sobel1,CV_16S,1,1,3);
convertScaleAbs(xuenai_sobel1,xuenai_sobel1);
imshow("xuenai_sobel1",xuenai_sobel1);

//在x,y方向上分别各调用一次Sobel
Mat xuenai_xsobel(xuenai.size(),xuenai.type());Mat xuenai_ysobel(xuenai.size(),xuenai.type());Mat xuenai_sobel2(xuenai.size(),xuenai.type());
Sobel(xuenai_gray,xuenai_xsobel,CV_16S,1,0,3);
convertScaleAbs(xuenai_xsobel,xuenai_xsobel);
Sobel(xuenai_gray,xuenai_ysobel,CV_16S,0,1,3);
convertScaleAbs(xuenai_ysobel,xuenai_ysobel);
addWeighted(xuenai_xsobel,0.5,xuenai_ysobel,0.5,0,xuenai_sobel2);
imshow("xuenai_sobel2",xuenai_sobel2);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/e3544bbf449d8a07c0bacc38d91d04ca.png)

**可以看到效果差了很多**

## 4.10 scharr（对灰度图）

### 4.10.1 卷积核形状（ksize恒定为3）

虽然Sobel算子可以有效的提取图像边缘，但是对图像中较弱的边缘提取效果较差。因此为了能够有效的提取出较弱的边缘，需要将像素值间的差距增大，因此引入Scharr算子。Scharr算子是对Sobel算子差异性的增强，因此两者之间的在检测图像边缘的原理和使用方式上相同。

```cpp
Mat kernalX=Mat_<int>(Size(3,3))<<(-3,0,3
                                    -10,0,10
                                    -3,0,3);
Mat kernalY=Mat_<int>(Size(3,3))<<(-3,-10,3
                                     0,0,0
                                     3,10,3);
```

### 4.10.2 API

```cpp
CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,
                          int dx, int dy, double scale = 1, double delta = 0,
                          int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

ddepth(destination depth)：输出图片的深度（CV_16F）

dx：x方向导数的阶数，一般取1

dy：y方向导数的阶数，一般取1

scale：生成图与原图的缩放比例，默认为1

delta：额外的增量，默认为0

borderType：边界填充方式，默认为黑边

### 4.10.3 流程

1. 用cvtColor函数转**灰度图**
2. **在x,y方向上分别各调用一次Scharr**
3. **用convertScaleAbs函数转换到CV_8U，否则无法显示**
4. **用addWeighted函数把两张输出图片加在一起**

## 4.11 Laplacian（对灰度图）

### 4.11.1 卷积核形状（ksize=3）

```cpp
Mat kernal=Mat_<int>(Size(3,3))<<(0,-1,0
                                   -1,4,-1
                                   0,-1,0);
```

Laplacian算子的卷积核形状决定了它 **对噪声非常敏感** ，因此，通常需要通过 **滤波平滑处理** 。

### 4.11.2 API

```cpp
CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                             int ksize = 1, double scale = 1, double delta = 0,
                             int borderType = BORDER_DEFAULT );
```

src(source)：输入图片

dst(destination)：输出图片

ddepth(destination depth)：输出图片的深度（CV_16F）

scale：生成图与原图的缩放比例，默认为1

delta：额外的增量，默认为0

borderType：边界填充方式，默认为黑边

### 4.11.3 流程

1. **用中值滤波等操作平滑处理**
2. 用cvtColor函数转**灰度图**
3. 用Laplacian函数处理
3. **用convertScaleAbs函数转换到CV_8U，否则无法显示**

## 4.12 Canny（recommended）

### 4.12.1 API

```cpp
CV_EXPORTS_W void Canny( InputArray image, OutputArray edges,
                         double threshold1, double threshold2,
                         int apertureSize = 3, bool L2gradient = false );
```

image：输入图片

edges：输出图片

threshold1：最小阈值

threshold2：最大阈值

**高于threshold2被认为是真边界，低于threshold1被抛弃，介于二者之间，则取决于是否与真边界相连。**

apertureSize：Sobel卷积核的大小，默认为3。 **核越大，对噪声越不敏感，但是边缘检测的错误也会随之增加** 。

L2gradient：计算图像梯度幅度的标识，默认为false，表示L1范数（直接将两个方向的导数的绝对值相加）。如果使用true，表示L2范数（两个方向的导数的平方和再开方）

### 4.12.2 流程

1. **用中值滤波等操作平滑处理**
2. 用Canny函数处理  **（不支持原地运算）**

### 4.12.3 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);

    Mat xuenai_canny(xuenai.size(),xuenai.type());
    Canny(xuenai,xuenai_canny,60,150);
    imshow("xuenai_canny",xuenai_canny);
    waitKet();
```

![](https://img-blog.csdnimg.cn/img_convert/e0509f034d9e5e414f1d1e0cf362540c.png)
## 4.13 添加噪声
为了检测算法的稳定性，常常需要在图片中人为地添加一些噪声来进行检验。
###  4.13.1 椒盐噪声

```cpp
static void addSaltNoise(const Mat& src,Mat& dst,int num=1000)
{
    dst=src.clone();
    for (int k = 0; k < num; k++)
    {
        //随机取值行列，得到像素点(i,j)
        int i = rand() % dst.rows;
        int j = rand() % dst.cols;
        //修改像素点(i,j)的像素值
        for(int channel=0;channel<src.channels();channel++){
            dst.ptr(i,j)[channel]=255;
        }
    }
    for (int k = 0; k < num; k++)
    {
        //随机取值行列
        default_random_engine engine;
        uniform_int_distribution<unsigned>u(0,10000);
        int i = rand() % dst.rows;
        int j = rand() % dst.cols;
        //修改像素点(i,j)的像素值
        for(int channel=0;channel<src.channels();channel++){
            dst.ptr(i,j)[channel]=0;
        }
    }
    return;
}
```
src(source)：输入图片
dst(destination)：输出图片
num(number)：噪声的个数
###  4.13.2 高斯噪声

```cpp
static void addGaussianNoise(const Mat& src,Mat& dst,InputArray meanValue=10,InputArray std=36){
    dst=src.clone();
    //构造高斯噪声矩阵
    Mat noise(dst.size(),dst.type());
    RNG rng(time(NULL));
    rng.fill(noise, RNG::NORMAL, meanValue, std);
    //将高斯噪声矩阵与原图像叠加得到含噪图像
    dst+=noise;
    return ;
}
```
src(source)：输入图片
dst(destination)：输出图片
meanValue：高斯函数均值
std(standard deviation)：高斯函数标准差
####  随机数填充矩阵

```cpp
void RNG::fill( InputOutputArray mat, int distType, InputArray a, InputArray b, bool saturateRange = false );
```
mat：输入输出矩阵，最多支持4通道，超过4通道先用reshape()改变结构
distType：可选UNIFORM 或 NORMAL，分别表示均匀分布和高斯分布
a：disType是UNIFORM,a表示下界(闭区间)；disType是NORMAL,a表示均值
b：disType是UNIFORM,b表示上界(开区间)；disType是NORMAL,b表示标准差
saturateRange：只针对均匀分布有效。当为真的时候，会先把产生随机数的范围变换到数据类型的范围，再产生随机数；如果为假，会先产生随机数，再进行截断到数据类型的有效区间。
# 5.画几何图形
## 5.1 直线
### 5.1.1 API

```cpp
CV_EXPORTS_W void line(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,int thickness = 1, int lineType = LINE_8, int shift = 0);
```

img(image)：输入图片

pt1(point1)：端点1

pt2(point2)：端点2

color：颜色

thickness：粗细

lineType：连通类型

shift：坐标点小数点位数(not important)

### 5.1.2连通类型

```cpp
enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};
```

* LINE_4与LINE_8差别不大，而LINE_AA的抗锯齿效果显著

## 5.2 正矩形

### 5.2.1API

```cpp
CV_EXPORTS_W void rectangle(InputOutputArray img, Point pt1, Point pt2,
                          const Scalar& color, int thickness = 1,
                          int lineType = LINE_8, int shift = 0);
```

img(image)：输入图片

pt1(point1)：左上角端点

pt2(point2)：右下角端点

color：颜色

thickness：粗细

lineType：连通类型

shift：坐标点小数点位数(not important)

```cpp
CV_EXPORTS_W void rectangle(InputOutputArray img, Rect rec,
                          const Scalar& color, int thickness = 1,
                          int lineType = LINE_8, int shift = 0);
```

img(image)：输入图片

rec(rect)：一个矩形

color：颜色

thickness：粗细

lineType：连通类型

shift：坐标点小数点位数(not important)

## 5.3 圆形

### 5.3.1 API

```cpp
CV_EXPORTS_W void circle(InputOutputArray img, Point center, int radius,
                       const Scalar& color, int thickness = 1,
                       int lineType = LINE_8, int shift = 0);
```

img(image)：输入图片

center：圆心坐标

radius：半径

color：颜色

**thickness：粗细。若取负值，则表示进行填充**

lineType：连通类型

shift：坐标点小数点位数(not important)

## 5.4 椭圆

### 5.4.1 API

```cpp
CV_EXPORTS_W void ellipse(InputOutputArray img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness = 1,
                        int lineType = LINE_8, int shift = 0);
```

img(image)：输入图片

center：圆心坐标

axes：（x方向上半轴长，y方向上半轴长）

angle：**顺时针偏角**

startAngle：以x方向上的半轴为起点，偏移一定角度后的起点，从此起点开始画椭圆

endAngle：以x方向上的半轴为起点，偏移一定角度后的终点，到此为止结束画椭圆

**确定起点和终点后，顺时针方向画**

color：颜色

**thickness：粗细。若取负值，则表示进行填充**

lineType：连通类型

shift：坐标点小数点位数(not important)

### 5.4.2 效果

```cpp
Mat canvas(Size(1000,1000),CV_8U,Scalar(255));
ellipse(canvas,Point2f(500,500),Size(50,100),0,0,90,Scalar(0,0,0),5);
imshow("canvas",canvas);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/94f1559f52fec20adc03fee8e954ece4.png)

```
Mat canvas(Size(1000,1000),CV_8U,Scalar(255));
ellipse(canvas,Point2f(500,500),Size(50,100),20,0,360,Scalar(0,0,0),5);
imshow("canvas",canvas);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/7ee8733f0daf524b918d560b71ff29b9.png)

```
Mat canvas(Size(1000,1000),CV_8U,Scalar(255));
ellipse(canvas,Point2f(500,500),Size(50,100),20,0,180,Scalar(0,0,0),5);
imshow("canvas",canvas);
waitKey();
```

![](https://img-blog.csdnimg.cn/img_convert/abecc8bdb3f811eaa659d42737dc8d32.png)

## 5.5 斜矩形

### 5.5.1 API(通过RotatedRect类和line函数实现)

```cpp
class CV_EXPORTS RotatedRect
{
public:
    //! default constructor
    RotatedRect();
   
    /*center：质心坐标
      size：（x方向上全边长，y方向上全边长）
      angle：顺时针偏角
    */
    RotatedRect(const Point2f& center, const Size2f& size, float angle);

    /**
    三点确定一矩形，记得要互相垂直
     */
    RotatedRect(const Point2f& point1, const Point2f& point2, const Point2f& point3);

    /** 返回四个角点坐标，要用Point2f类型的数组对象作为参数传入，不能是仅仅是Point类型的数组对象*/
    void points(Point2f pts[]) const;

    //! returns the minimal up-right integer rectangle containing the rotated rectangle
    Rect boundingRect() const;

    //! returns the minimal (exact) floating point rectangle containing the rotated rectangle, not intended for use with images
    Rect_<float> boundingRect2f() const;

    //! returns the rectangle mass center
    Point2f center;

    //! returns width and height of the rectangle
    Size2f size;

    //! returns the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    float angle;
};
```

下面是自定义的一个快捷画斜矩形的函数

```cpp
void drawRotatedRect(InputOutputArray img, RotatedRect rRect,const Scalar& color, int thickness = 1,int lineType = LINE_8, int shift = 0){
    Point2f vertices[4];
    rRect.points(vertices);
    for(int i=0;i<4;i++){
        line(img,vertices[i],vertices[(i+1)%4],color,lineType,shift);
    }
}
```

# 6.Trackbar控件

## 6.1 createTrackbar创建滚动条

### 6.1.1 API

```cpp
CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);
```

trackbarname：滚动条名字

winname(window name)：窗体名字。**要先用nameWindow创建好同名窗体，滚动条才会出现**

value：欲控制的**变量的地址**

count：欲控制的变量的最大值（最小为0）

onChange：回调函数，默认为空。如果想要传入，那么其 **参数是固定的** 。

```cpp
void onChange(int,void*);
```

userdata：万能指针，默认为空。如果想要传入，通常用一个类的对象的地址。

## 6.2 getTrackbarPos获得滚动条当前的值

```cpp
CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);
```

trackbarname：滚动条名字

winname(window name)：窗体名字

## 6.3 使用方式一(recommended)

### 6.3.1 原理

不使用createTrackbar函数的参数value、onChange、userdata参数。**通过while(1)的无限循环**，在循环中不断地用getTrackbarPos函数**动态地获取**滚动条的值，然后**在循环内部**用这些值进行操作。

### 6.3.2 效果

```cpp
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);

    namedWindow("xuenai_rotate");
    Mat xuenai_rotate(xuenai.size(), xuenai.type());
    createTrackbar("angle","xuenai_rotate", nullptr,360);
    while (1) {
        int angle= getTrackbarPos("angle","xuenai_rotate");
        Mat M = getRotationMatrix2D(Point2f(xuenai.cols / 2, xuenai.rows / 2), angle, 1);
        warpAffine(xuenai, xuenai_rotate, M, xuenai.size());
        imshow("xuenai_rotate",xuenai_rotate);
        waitKey(20);
    }
```

![](https://img-blog.csdnimg.cn/img_convert/a221a77f99d1449d72cae400f3faea8d.png)

![](https://img-blog.csdnimg.cn/img_convert/16b2444105c18183ac864cc02c7b976e.png)

## 6.4 使用方式二

### 6.4.1 原理

不使用getTrackbarPos函数，使用createTrackbar的全部参数，**在onChange回调函数中完成所有操作**，由于回调函数的参数表是固定的，因此**需要userdata传入所需数据。**在每次移动滚动条时，**相当于调用了一次回调函数**，就完成了操作。**结尾没有waitKey(0)就显示不了多久。**

### 6.4.2 效果

```cpp
class TrackbarUserdata{
public:
    Mat input;
    Mat output;
    int angle=0;
    string winname;
};

void RotateonChange(int,void *userdata) {

    TrackbarUserdata *data = (TrackbarUserdata *) userdata;
    int rows = data->input.rows;
    int cols = data->output.cols;
    Mat M = getRotationMatrix2D(Point2f(rows / 2, cols / 2), data->angle, 1);
    warpAffine(data->input,data->output,M,data->input.size());
    imshow(data->winname,data->output);
    waitKey(10);

}
int main(){
    Mat xuenai = imread("xuenai.jpg");
    imshow("xuenai",xuenai);

    Mat xuenai_rotate(xuenai.size(), xuenai.type());
    
    TrackbarUserdata userdata;
    userdata.input=xuenai;
    userdata.output=xuenai_rotate;
    userdata.winname="xuenai_rotate";
    namedWindow(userdata.winname);
    createTrackbar("angle",userdata.winname, &userdata.angle,360, RotateonChange,&userdata);
    waitKey();
    
    return 0;
}
```

![](https://img-blog.csdnimg.cn/img_convert/1755517361ad73efe5aa8281fcdbfcd4.png)

![](https://img-blog.csdnimg.cn/img_convert/bccd07ac8de3722cd5cc6e6ac04b9933.png)

# 7.轮廓检测（一般对二值图）

## 7.1 查找轮廓

### 7.1.1 API

```cpp
CV_EXPORTS_W void findContours( InputArray image, OutputArrayOfArrays contours,
                              OutputArray hierarchy, int mode,
                              int method, Point offset = Point());

/** @overload */
CV_EXPORTS void findContours( InputArray image, OutputArrayOfArrays contours,
                              int mode, int method, Point offset = Point());
```

image：输入图片

contours：保存输出轮廓的点坐标。**通常用vector<vector<Point>>数据类型担任，通过Point可以看出存储的是坐标。**

hierarchy：可选参数，保存输出轮廓的层级关系。**通常用vector<Vec4i>数据类型担任。**

**hierarchy[i][0]：第i个轮廓的同一层级后一个轮廓的索引编号。**

**hierarchy[i][1]：第i个轮廓的同一层级前一个轮廓的索引编号。**

**hierarchy[i][2]：第i个轮廓的子轮廓的索引编号。**

**hierarchy[i][3]：第i个轮廓的父轮廓的索引编号。**

如果当前轮廓没有对应的后一个轮廓、前一个轮廓、父轮廓或内嵌轮廓的话，则hierarchy[i][0] ~hierarchy[i][3]的相应位被设置为默认值-1。

mode：轮廓层级的检测模式。

method：轮廓坐标点的储存方式

offset：额外偏移量，**在每一个检测出的轮廓点上加上该偏移量，可以是负值。当所分析图像是另外一个图像的ROI的时候，通过加减这个偏移量，可以把ROI图像的检测结果投影到原始图像对应位置上。**

### 7.1.2 轮廓层级检测模式：索引号（层级）

```cpp
enum RetrievalModes {
    RETR_EXTERNAL  = 0,
    RETR_LIST      = 1,
    RETR_CCOMP     = 2,
    RETR_TREE      = 3,
};
```

#### RETR_EXTERNAL（索引顺序：从右下到左上）

只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
![在这里插入图片描述](https://img-blog.csdnimg.cn/d29a060b4cf14dc8872e1f38bb63aeb2.jpeg#pic_center)


#### RETR_LIST(recommended)（索引顺序：从右下到左上，由外到内）

检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立层级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓，所以hierarch[i]向量内所有元素的第3、第4个分量都会被置为-1。

![](https://img-blog.csdnimg.cn/img_convert/9420b0b5d69ce6d404a556896f94eed0.jpeg)

#### RETR_CCOMP(not recommended)（索引顺序：由内到外，从右下到左上）

检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层

![在这里插入图片描述](https://img-blog.csdnimg.cn/90b1e5ea29814672b6dc37bcd0844d3b.jpeg#pic_center)


#### RETR_TREE(recommended)

检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。

![在这里插入图片描述](https://img-blog.csdnimg.cn/31d9ba62e87c45f6b99068091e7b6daa.jpeg#pic_center)


### 7.1.3 轮廓坐标点储存方式

```
enum ContourApproximationModes {
    CHAIN_APPROX_NONE      = 1,
    CHAIN_APPROX_SIMPLE    = 2,
    CHAIN_APPROX_TC89_L1   = 3,
    CHAIN_APPROX_TC89_KCOS = 4
};
```

CHAIN_APPROX_NONE：保存物体边界上所有连续的轮廓点到contours向量内

CHAIN_APPROX_SIMPLE(recommended)：仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，**拐点与拐点之间直线段上的信息点不予保留，效率比较高。**

CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS：使用tehChinl chain 近似算法(not important)

## 7.2 绘制轮廓

### 7.2.1 API

```cpp
CV_EXPORTS_W void drawContours( InputOutputArray image, InputArrayOfArrays contours,
                              int contourIdx, const Scalar& color,
                              int thickness = 1, int lineType = LINE_8,
                              InputArray hierarchy = noArray(),
                              int maxLevel = INT_MAX, Point offset = Point() );
```

image：输入图片

contours：输入轮廓数组

contourIdx(contour index)：欲绘制的轮廓的索引值，输入-1可以绘制所有轮廓

color：绘制颜色

thickness：线条粗细，默认为1。输入-1则表示填充。

lineType：连通类型。

hierarchy：可选的层次结构信息。它仅在当你需要绘制一些轮廓线时被使用。（详见参数maxLevel）默认为noArray(),返回一个空数组。

maxLevel：绘制轮廓线的最高级别。此参数仅在参数hierarchy有效时被考虑。

* 如果为0，只有被指定的轮廓被绘制。
* 如果为1，绘制被指定的轮廓和 **其下一级轮廓** 。
* 如果为2，绘制被指定的轮廓和 **其所有子轮廓** 。

offset：额外偏移量。

## 7.3 轮廓面积和周长

### 7.3.1 面积（非原地算法）

```cpp
CV_EXPORTS_W double contourArea( InputArray contour, bool oriented = false );
```

contour：某**一个**轮廓，数据类型vector\<Point\>

oriented: 有方向的区域标志。(not important)

1. true: 此函数依赖轮廓的方向（顺时针或逆时针）返回一个已标记区域的值。
2. false: 默认值。意味着返回不带方向的绝对值。

**此函数利用格林公式计算轮廓的面积。对于具有自交点的轮廓，该函数几乎肯定会给出错误的结果。**

### 7.3.2周长（非原地算法）

```cpp
CV_EXPORTS_W double arcLength( InputArray curve, bool closed );
```

curve：某**一个**轮廓，数据类型vector\<Point\>

closed：轮廓是否是闭合的。

## 7.4 多边形逼近

```cpp
CV_EXPORTS_W void approxPolyDP( InputArray curve,
                                OutputArray approxCurve,
                                double epsilon, bool closed );
```

curve：某**一个**轮廓，数据类型vector\<Point\>

approxCurve：输出多边形的点集，数据类型vector<Point>

epsilon：设置精度，越小则精度越高，多边形越趋近于曲线，拟合效果更好但效率低。

closed：轮廓是否是闭合的。

## 7.5 凸包

```cpp
CV_EXPORTS_W void convexHull( InputArray points, OutputArray hull,
                              bool clockwise = false, bool returnPoints = true );
```

points：输入点集

hull：输出凸包。 **数据类型取决于returnPoints，vector\<Point\>或vector\<int\>**

clockwise：拟合凸包的直线的转动方向，TRUE为顺时针，否则为逆时针。

returnPoints：若为true，则在hull中存储点的坐标。若为false，则在hull中存储点的索引，索引值根据参数points得到。默认为true

## 7.6 外接矩形

### 7.6.1最小外接矩形（返回RotatedRect）

```cpp
CV_EXPORTS_W RotatedRect minAreaRect( InputArray points );
```

points：输入点集

### 7.6.2最大外界矩形（返回Rect）

```
CV_EXPORTS_W Rect boundingRect( InputArray array );
```

points：输入点集
#  8.特征工程
##  8.1 模板匹配
###  8.1.1 原理 
模板图像在原图像上从原点开始移动，计算**模板与原图被模板覆盖的地方**的差别程度，计算方法有几种，然后将每次计算的结果放进输出矩阵。若原图像为A\*B大小，模板为a\*b大小，则 **输出矩阵为(A-a+1)*(B-b+1)** 大小。
###  8.1.2 API

```cpp
CV_EXPORTS_W void matchTemplate( InputArray image, InputArray templ,
                                 OutputArray result, int method, InputArray mask = noArray() );
```
image：输入图像

templ(template)：模板图像

result：输出矩阵，**深度为CV_32FC1**。若原图像为A\*B大小，模板为a\*b大小，则 **输出矩阵为(A-a+1)*(B-b+1)** 大小。

method：模板匹配计算方法。

mask：掩码图像。**其大小与模板图像必须相同，且必须为灰度图**。匹配时，对于掩码中的非0像素匹配算法起作用，掩码中的灰度值为0的像素位置，匹配算法不起作用。
###  8.1.3 模板匹配计算方法

```cpp
enum TemplateMatchModes {
    TM_SQDIFF        = 0, 
    TM_SQDIFF_NORMED = 1, 
    TM_CCORR         = 2, 
    TM_CCORR_NORMED  = 3, 
    TM_CCOEFF        = 4, 
    TM_CCOEFF_NORMED = 5 
};

```
TM_SQDIFF：计算平方误差，计算出来的值越小，则匹配得越好
TM_CCORR：计算相关性，计算出来的值越大，则匹配得越好
TM_CCOEFF：计算相关系数，计算出来的值越大，则匹配得越好
TM_SQDIFF_NORMED：计算归一化平方误差，计算出来的值越接近0，则匹配得越好
TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，则匹配得越好
TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，则匹配得越好
![在这里插入图片描述](https://img-blog.csdnimg.cn/16c0a3fe93e740e98bb3318eef303a96.png#pic_center)
###  8.1.4 掩码的使用
在进行特征匹配时，我们有时并不需要用整个图片作为模板，**因为模板的背景可能会干扰匹配的结果**。因此，我们需要加入掩码，就可以屏蔽掉背景进行模板匹配
####  获得掩码

 1. 模板图像转灰度图
 2. 二值化屏蔽背景
###  8.1.5 效果

```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai",xuenai);
        
        Mat templ= imread("xuenai_rect.jpg");
        imshow("template",templ);
        Mat match_result;
        matchTemplate(xuenai,templ,match_result,TM_SQDIFF);

        Point temLoc;
        Point minLoc;
        Point maxLoc;
        double min,max;
        minMaxLoc(match_result,&min,&max,&minLoc,&maxLoc);
        temLoc=minLoc;

        rectangle(xuenai,Rect(temLoc.x,temLoc.y,templ.cols,templ.rows),Scalar(0,0,255));
        imshow("xuenai_match",xuenai);
        waitKey();
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/906b5f8c8b4446eca64b3f9883b6188c.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8be117c0005b4843bd1e480a09cc7f7d.png#pic_center)



###  8.1.5 模板匹配的缺陷
####  无法应对旋转

```cpp
        Mat xuenai = imread("xuenai.jpg");
        rotate(xuenai,xuenai,ROTATE_90_CLOCKWISE);
        imshow("xuenai",xuenai);

        Mat templ= imread("xuenai_rect.jpg");
        Mat match_result;
        matchTemplate(xuenai,templ,match_result,TM_SQDIFF);

        Point temLoc;
        Point minLoc;
        Point maxLoc;
        double min,max;
        minMaxLoc(match_result,&min,&max,&minLoc,&maxLoc);
        temLoc=minLoc;

        rectangle(xuenai,Rect(temLoc.x,temLoc.y,templ.cols,templ.rows),Scalar(0,0,255));
        imshow("xuenai_match",xuenai);
        waitKey();
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/f4c272690da049688bea5bbd14db6a22.png#pic_center)


####  无法应对缩放

```cpp
        Mat xuenai = imread("xuenai.jpg");
        resize(xuenai,xuenai,Size(500,500));
        imshow("xuenai",xuenai);

        Mat templ= imread("xuenai_rect.jpg");
        Mat match_result;
        matchTemplate(xuenai,templ,match_result,TM_SQDIFF);

        Point temLoc;
        Point minLoc;
        Point maxLoc;
        double min,max;
        minMaxLoc(match_result,&min,&max,&minLoc,&maxLoc);
        temLoc=minLoc;

        rectangle(xuenai,Rect(temLoc.x,temLoc.y,templ.cols,templ.rows),Scalar(0,0,255));
        imshow("xuenai_match",xuenai);
        waitKey();
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/a7a6d393150147b3916fc8450633232d.png#pic_center)
##  8.2 cornerHarris（对灰度图）
###  8.2.1 角点的描述

 - 一阶导数(即灰度的梯度)的局部最大所对应的像素点；
 - 两条及两条以上边缘的交点；
 - 图像中梯度值和梯度方向的变化速率都很高的点；
 - 角点处的一阶导数最大，二阶导数为零，指示物体边缘变化不连续的方向。

###  8.2.2 原理（前置知识要求：线性代数）(bolcksize=2的情况)
使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。
考虑到一个灰度图像 . 划动窗口  (with displacements  在x方向和  方向)  计算像素灰度变化。

![在这里插入图片描述](https://img-blog.csdnimg.cn/a4ba8fb091bf40fc84a5e206b522d237.png#pic_center)


其中:
- w(x,y) is the window at position (x,y)
- I(x,y) is the intensity at (x,y)
- I(x+u,y+v) is the intensity at the moved window (x+u,y+v)、

为了寻找带角点的窗口，搜索像素灰度变化较大的窗口。于是, 我们期望最大化以下式子:
![在这里插入图片描述](https://img-blog.csdnimg.cn/c2defb22ed5240a4a57cd7535956b97a.png#pic_center)
泰勒展开:
![在这里插入图片描述](https://img-blog.csdnimg.cn/a2612fdd83e547e2bb4fa0c22205a6bb.png#pic_center)

 - Ix，Iy是通过**sobel算子**计算的一阶导数

矩阵化:
![在这里插入图片描述](https://img-blog.csdnimg.cn/059f87c8f0c8499bb1cabdbeba91023b.png#pic_center)
得二次型:
![在这里插入图片描述](https://img-blog.csdnimg.cn/28b99fb274df4ec8ba89c7b92432c150.png#pic_center)
因此有等式:
![在这里插入图片描述](https://img-blog.csdnimg.cn/907f9a4b10c14ab792cddf69273e6c87.png#pic_center)
每个窗口中计算得到一个值。这个值决定了这个窗口中是否包含了角点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/9224ab1492c748b1b3bc9cf879feadc7.png#pic_center)

其中，det(M) = 矩阵M的行列式，trace(M) = 矩阵M的迹
- R为正值时，检测到的是角点，R为负时检测到的是边，R很小时检测到的是平坦区域。
###  8.2.3 API

```cpp
CV_EXPORTS_W void cornerHarris( InputArray src, OutputArray dst, int blockSize,
                                int ksize, double k,
                                int borderType = BORDER_DEFAULT );
```
src(source)：输入图片 **（灰度图）**，**深度要求：CV_8UC1或CV_32FC1**

dst(destination)：输出图片

bolckSize：检测窗口的大小，**越大则对角点越敏感，一般取2**

ksize(kernal size)：使用sobel算子计算一阶导数时的滤波器大小，一般取3即可。

k：计算用到的系数，**公认一般取值在0.02~0.06。**

borderType ：边界填充方式，默认为黑边。

###  8.2.4 流程

 1. 转灰度图
 2. 使用cornerHarris函数检测
 3. **使用normalize函数归一化处理和convertScaleAbs绝对化**
 4. 遍历输出图像并筛选角点。**不要使用迭代器的遍历方式，因为太慢！**
 
- 经过实测，以下这种**用行数调用ptr函数**的遍历方式是最快的
```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai", xuenai);

//转灰度图
        Mat xuenai_gray(xuenai.size(),xuenai.type());
        cvtColor(xuenai,xuenai_gray,COLOR_BGR2GRAY);

        Mat xuenai_harris;
        cornerHarris(xuenai_gray,xuenai_harris,2,3,0.04);
        normalize(xuenai_harris,xuenai_harris,0,255,NORM_MINMAX,-1);
        convertScaleAbs(xuenai_harris,xuenai_harris);

        namedWindow("xuenai_harris");
        createTrackbar("threshold","xuenai_harris", nullptr,255);

        while (1) {
            int thres = getTrackbarPos("threshold", "xuenai_harris");
            if(thres==0)thres=100;
            Mat harris_result=xuenai.clone();
            for(int i=0;i<xuenai_harris.rows;i++){
                uchar * ptr =xuenai_harris.ptr(i);
                for(int j=0;j<xuenai_harris.cols;j++){
                    int value=(int) *ptr;
                    if(value>thres){
                        circle(harris_result, Point(j,i), 3, Scalar(0, 0, 255));
                    }
                    ptr++;
                }
            }
            imshow("xuenai_harris",harris_result);
            if (waitKey(0) == 'q')break;
        }
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/16d050f0065d4586b3b70bcfc00e9baf.png#pic_center)
###  8.2.5 优点与缺点
####  测试代码
```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai", xuenai);
        namedWindow("panel");
        createTrackbar("threshold","panel", nullptr,255);
        createTrackbar("angle","panel", nullptr,360);
        createTrackbar("width","panel", nullptr,1000);
        createTrackbar("height","panel", nullptr,1000);

        while (1) {
            int thres = getTrackbarPos("threshold", "panel");
            if(thres==0)thres=100;
            int width = getTrackbarPos("width", "panel");
            if(width==0)width=xuenai.cols;
            int height = getTrackbarPos("height", "panel");
            if(height==0)height=xuenai.rows;
            int angle = getTrackbarPos("angle","panel");
            

            Mat xuenai_harris, xuenai_transform=xuenai.clone();

            resize(xuenai_transform,xuenai_transform,Size(width,height));

            Mat M= getRotationMatrix2D(Point2f(xuenai.cols/2,xuenai.rows/2),angle,1);
            warpAffine(xuenai_transform,xuenai_transform,M,xuenai_transform.size());

            Mat xuenai_gray(xuenai.size(),xuenai.type());
            cvtColor(xuenai_transform,xuenai_gray,COLOR_BGR2GRAY);

            cornerHarris(xuenai_gray,xuenai_harris,2,3,0.04);
            normalize(xuenai_harris,xuenai_harris,0,255,NORM_MINMAX,-1);
            convertScaleAbs(xuenai_harris,xuenai_harris);

            Mat harris_result=xuenai_transform.clone();
            for(int i=0;i<xuenai_harris.rows;i++){
                uchar * ptr =xuenai_harris.ptr(i);
                for(int j=0;j<xuenai_harris.cols;j++){
                    int value=(int) *ptr;
                    if(value>thres){
                        circle(harris_result, Point(j,i), 3, Scalar(0, 0, 255));
                    }
                    ptr++;
                }
            }
            imshow("xuenai_harris",harris_result);
            if (waitKey(0) == 'q')break;
        }
   ```


####  图片旋转，角点不变
![在这里插入图片描述](https://img-blog.csdnimg.cn/3023822133ad489fbcca66b8a692544e.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f7bcbf5e918d4488bc08310ab5e89760.png#pic_center)

####  图片缩放，角点改变

![在这里插入图片描述](https://img-blog.csdnimg.cn/bf2c05b6a7f1425893df5c0dba046a9f.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fa1d6fa2b4b34a4bb64bb670cbe46521.png#pic_center)

##  8.3  Shi-Tomasi（对灰度图）
###  8.3.1 原理
由于cornerHarris角点检的稳定性与k密切相关，而k是个经验值，难以设定最佳值，Shi-Tomasi在这一点上进行了改进
- 计算角点分数
![在这里插入图片描述](https://img-blog.csdnimg.cn/7574b7b25aae46dfa6fbbaca1516054c.png#pic_center)
###  8.3.2 API

```cpp
CV_EXPORTS_W void goodFeaturesToTrack( InputArray image, OutputArray corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray mask = noArray(), int blockSize = 3,
                                     bool useHarrisDetector = false, double k = 0.04 );

CV_EXPORTS_W void goodFeaturesToTrack( InputArray image, OutputArray corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray mask, int blockSize,
                                     int gradientSize, bool useHarrisDetector = false,
                                     double k = 0.04 );
```
image：输入图片

corners：输出角点的点集，数据类型vector<Point2f>

maxCorners：控制输出角点点集的上限个数，即控制corners.size()。**输入0则表示不限制上限**

qualityLevel：质量系数（小于1.0的正数，一般在0.01-0.1之间），**表示可接受角点的最低质量水平**。该系数乘以输入图像中最大的角点分数，作为可接受的最小分数；例如，如果输入图像中最大的角点分数值为1500且质量系数为0.01，那么所有角点分数小于15的角都将被忽略。

minDistance：角点之间的最小欧式距离，**小于此距离的点将被忽略**。

mask：掩码图像。**其大小与输入图像必须相同，且必须为灰度图**。计算时，对于掩码中的非0像素算法起作用，掩码中的灰度值为0的像素位置，算法不起作用。

blockSize：检测窗口的大小，**越大则对角点越敏感**。

useHarrisDetector：用于指定角点检测的方法，如果是true则使用Harris角点检测，false则使用Shi Tomasi算法。默认为False。

k：默认为0.04，只有useHarrisDetector参数为true时起作用。
###  8.3.3 流程

 1. 转灰度图
 2. 使用Shi-Tomasi函数检测
 3. 遍历角点集合即可

###  8.3.4 效果

 - Shi-Tomasi同样具有旋转不变性和尺度可变性
 

```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai", xuenai);
        namedWindow("panel");
        createTrackbar("threshold","panel", nullptr,255);
        createTrackbar("angle","panel", nullptr,360);
        createTrackbar("width","panel", nullptr,1000);
        createTrackbar("height","panel", nullptr,1000);

        while (1) {
            int thres = getTrackbarPos("threshold", "panel");
            if(thres==0)thres=100;
            int width = getTrackbarPos("width", "panel");
            if(width==0)width=xuenai.cols;
            int height = getTrackbarPos("height", "panel");
            if(height==0)height=xuenai.rows;
            int angle = getTrackbarPos("angle","panel");

            Mat xuenai_transform=xuenai.clone();

            resize(xuenai_transform,xuenai_transform,Size(width,height));

            Mat M= getRotationMatrix2D(Point2f(xuenai.cols/2,xuenai.rows/2),angle,1);
            warpAffine(xuenai_transform,xuenai_transform,M,xuenai_transform.size());

            Mat xuenai_gray(xuenai.size(),xuenai.type());
            cvtColor(xuenai_transform,xuenai_gray,COLOR_BGR2GRAY);

            vector<Point2f>xuenai_cornersSet;
            goodFeaturesToTrack(xuenai_gray,xuenai_cornersSet,0,0.1,10);
            for(auto corner:xuenai_cornersSet){
                circle(xuenai_transform,corner,3,Scalar(0,0,255));
            }

            imshow("xuenai_corners",xuenai_transform);
            if (waitKey(0) == 'q')break;
        }
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/e11d955293d54951935325892819fc3d.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b4b2ac33bc1243db8e1b7dc111defe3f.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2ac71b6ab89542a793651284f9adc506.png#pic_center)
##  8.4 SIFT与SURF（对灰度图）
###  8.4.1 概述
cornerHarris和Shi-Tomasi都没能保证角点在尺度上的稳定性，因此**SIFT和SURF针对这一特点进行了优化**。由于其数学原理较为复杂，请自行查阅相关论文和文献，本文不再赘述。
相较于cornerHarris和Shi-Tomasi，SIFT和SURF的优点是显著的，其检测出的角点**对旋转、尺度缩放、亮度变化等保持不变性，对视角变换、仿射变化、噪声也保持一定程度的稳定性**，是一种非常优秀的局部特征描述算法。
需要注意的是，SIFT和SURF的计算量较为庞大，**难以做到实时运算**。SIFT和SURF两者相比，**SIFT更为精确，SURF更为高效。**
###  8.4.2 API
####  构造函数
```cpp
    CV_WRAP static Ptr<SIFT> SIFT::create(int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6);
        
    CV_WRAP static Ptr<SIFT> SIFT::create(int nfeatures, int nOctaveLayers,
        double contrastThreshold, double edgeThreshold,
        double sigma, int descriptorType);
        
    CV_WRAP static Ptr<SURF> SURF::create(double hessianThreshold=100,
                  int nOctaves = 4, int nOctaveLayers = 3,
                  bool extended = false, bool upright = false);
```

 - 构造函数的参数设计复杂的数学原理，在此不进行解释，在使用时进行默认的构造即可。
####  进行关键点检测和描述子计算

```cpp
    CV_WRAP virtual void Feature2D::detect( InputArray image,
                                 CV_OUT std::vector<KeyPoint>& keypoints,
                                 InputArray mask=noArray() );
                                 
    CV_WRAP virtual void Feature2D::detect( InputArrayOfArrays images,
                         CV_OUT std::vector<std::vector<KeyPoint> >& keypoints,
                         InputArrayOfArrays masks=noArray() );
```
image：输入图像

keypoints：含多个关键点的vector\<KeyPoint\>。**使用detect时作为输出，使用compute时作为输入，使用detectAndCompute时可以作为输入也可以作为输出。**

mask：掩码图像。**其大小与输入图像必须相同，且必须为灰度图**。计算时，对于掩码中的非0像素算法起作用，掩码中的灰度值为0的像素位置，算法不起作用。
```cpp
    CV_WRAP virtual void Feature2D::compute( InputArray image,
                                  CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors );
    CV_WRAP virtual void Feature2D::compute( InputArrayOfArrays images,
                          CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> >& keypoints,
                          OutputArrayOfArrays descriptors );
    CV_WRAP virtual void Feature2D::detectAndCompute( InputArray image, InputArray mask,
                                           CV_OUT std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors,
                                           bool useProvidedKeypoints=false );
```
image：输入图像

keypoints：含多个关键点的vector\<KeyPoint\>。**使用detect时作为输出，使用compute时作为输入，使用detectAndCompute时可以作为输入也可以作为输出。**

descriptors：描述子，数据类型Mat，。**在进行特征匹配的时候会用到**。

useProvidedKeypoints：**false时，keypoints作为输出，并根据keypoints算出descriptors。true时，keypoints作为输入，不再进行detect，即不修改keypoints，并根据keypoints算出descriptors。**

####  drawKeypoints绘制关键点

```cpp
CV_EXPORTS_W void drawKeypoints( InputArray image, const std::vector<KeyPoint>& keypoints, InputOutputArray outImage,
                               const Scalar& color=Scalar::all(-1), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );
enum struct DrawMatchesFlags
{
  DEFAULT = 0, //!< Output image matrix will be created (Mat::create),
               //!< i.e. existing memory of output image may be reused.
               //!< Two source image, matches and single keypoints will be drawn.
               //!< For each keypoint only the center point will be drawn (without
               //!< the circle around keypoint with keypoint size and orientation).
  DRAW_OVER_OUTIMG = 1, //!< Output image matrix will not be created (Mat::create).
                        //!< Matches will be drawn on existing content of output image.
  NOT_DRAW_SINGLE_POINTS = 2, //!< Single keypoints will not be drawn.
  DRAW_RICH_KEYPOINTS = 4 //!< For each keypoint the circle around keypoint with keypoint size and
                          //!< orientation will be drawn.
};
```
image：输入图像

keypoints：含多个关键点的vector数组。

outImage：输出图像

color：绘制颜色信息，默认绘制的是随机彩色。

flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制。

DrawMatchesFlags::DEFAULT：只绘制特征点的坐标点，显示在图像上就是一个个小圆点，每个小圆点的圆心坐标都是特征点的坐标。

DrawMatchesFlags::DRAW_OVER_OUTIMG：函数不创建输出的图像，而是直接在输出图像变量空间绘制，要求本身输出图像变量就是一个初始化好了的，size与type都是已经初始化好的变量。

DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制。

DrawMatchesFlags::DRAW_RICH_KEYPOINTS ：绘制特征点的时候绘制的是一个个带有方向的圆，这种方法同时显示图像的坐标，size和方向，是最能显示特征的一种绘制方式。


###  8.4.3 流程

 1. 实例化SIFT或SURF对象
 2. **将输入图像转灰度图**
 3. 根据需要，调用detect函数或compute函数或detectAndCompute函数，检测关键点和计算描述子
 4. 调用drawKeypoints函数绘制关键点

###  8.4.4 效果

```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai", xuenai);
        namedWindow("panel");
        createTrackbar("threshold","panel", nullptr,255);
        createTrackbar("angle","panel", nullptr,360);
        createTrackbar("width","panel", nullptr,1000);
        createTrackbar("height","panel", nullptr,1000);

        while (1) {
            int thres = getTrackbarPos("threshold", "panel");
            if(thres==0)thres=100;
            int width = getTrackbarPos("width", "panel");
            if(width==0)width=xuenai.cols;
            int height = getTrackbarPos("height", "panel");
            if(height==0)height=xuenai.rows;
            int angle = getTrackbarPos("angle","panel");

            Mat xuenai_transform=xuenai.clone();

            resize(xuenai_transform,xuenai_transform,Size(width,height));

            Mat M= getRotationMatrix2D(Point2f(xuenai_transform.cols/2,xuenai_transform.rows/2),angle,1);
            warpAffine(xuenai_transform,xuenai_transform,M,xuenai_transform.size());

            Mat xuenai_gray(xuenai.size(),xuenai.type());
            cvtColor(xuenai_transform,xuenai_gray,COLOR_BGR2GRAY);

            Ptr<SIFT> sift=SIFT::create();
            Ptr<SURF> surf=SURF::create();
            vector<KeyPoint>xuenai_SiftKp,xuenai_Surfp;
            sift->detect(xuenai_gray,xuenai_SiftKp);
            surf->detect(xuenai_gray,xuenai_Surfp);

            Mat sift_result=xuenai_transform.clone(),surf_result=xuenai_transform.clone();
            drawKeypoints(sift_result,xuenai_SiftKp,sift_result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            drawKeypoints(surf_result,xuenai_Surfp,surf_result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imshow("sift_result",sift_result);
            imshow("surf_result",surf_result);
            if (waitKey(0) == 'q')break;
        }
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/e7e177a72bce448c834c8e8781cc9580.png#pic_center)
####  进行缩放和旋转
![在这里插入图片描述](https://img-blog.csdnimg.cn/e363923230c9424db30bc8443b67a34b.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3eff2dacd5a14791aecfe8340405ef50.png#pic_center)

 - 可以看到，无论是旋转还是缩放，关键点都保持得非常稳定。

##  8.5 FAST到OBR（对灰度图）
###  8.5.1 概述
前文已经阐述，SIFT和SURF已经做到了角点在旋转和缩放下的稳定性，但是它们还有一个致命的缺陷，**就是它们难以做到实时运算**，因此，FAST和OBR应运而生了。
####  FAST原理
从图片中选取一个坐标点P,获取该点的像素值,接下来判定该点是否为特征点.
选取一个以选取点P坐标为圆心的半径等于r的Bresenham圆(一个计算圆的轨迹的离散算法,得到整数级的圆的轨迹点),一般来说,这个圆上有16个点,如下所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/40f0ba43a7d749a8891414754dfa157e.png#pic_center)

 1. p在图像中表示一个被识别为兴趣点的像素。令它的强度为 Ip； 
 2. 选择一个合适的阈值t； 
 3. 考虑被测像素周围的16个像素的圆圈。 如果这16个像素中存在一组ñ个连续的像素的像素值，比 Ip+t 大，或比 Ip−t小，则像素p是一个角点。ñ被设置为12。 
 4. 使用一种快速测试（high-speed test）可快速排除了大量的非角点。这个方法只检测在1、9、5、13个四个位置的像素，（首先检测1、9位置的像素与阈值比是否太亮或太暗，如果是，则检查5、13）。如果p是一个角点，则至少有3个像素比 Ip+t大或比 Ip−t暗。如果这两者都不是这样的话，那么p就不能成为一个角点。然后可以通过检查圆中的所有像素，将全部分段测试标准应用于通过的对候选的角点。这种探测器本身表现出很高的性能，但有一些缺点： 
- 它不能拒绝n <12的候选角点。当n<12时可能会有较多的候选角点出现 
- 检测到的角点不是最优的，因为它的效率取决于问题的排序和角点的分布。 
- 角点分析的结果被扔掉了。过度依赖于阈值
- 多个特征点容易挤到一起。 
- 前三点是用机器学习方法解决的。最后一个是使用非极大值抑制来解决。具体不再展开。

**FAST算法虽然很快，但是没有建立关键点的描述子，也就无法进行特征匹配**
####  OBR简介
ORB 是 Oriented Fast and Rotated Brief 的简称，从这个简介就可以看出，OBR算法是基础FAST算法的改进。其中，Fast 和 Brief 分别是特征检测算法和向量创建算法。ORB 首先会从图像中查找特殊区域，称为关键点。关键点即图像中突出的小区域，比如角点，比如它们具有像素值急剧的从浅色变为深色的特征。然后 ORB 会为每个关键点计算相应的特征向量。ORB 算法创建的特征向量只包含 1 和 0，称为二元特征向量。1 和 0 的顺序会根据特定关键点和其周围的像素区域而变化。该向量表示关键点周围的强度模式，因此多个特征向量可以用来识别更大的区域，甚至图像中的特定对象。
关于Brief算法的具体原理本文不再赘述，请自行查阅相关论文和文献。

###  8.5.2 API
####  构造函数
```cpp
    CV_WRAP static Ptr<FastFeatureDetector> create( int threshold=10,
                                                    bool nonmaxSuppression=true,
                                                    FastFeatureDetector::DetectorType type=FastFeatureDetector::TYPE_9_16 );
                                                    
    CV_WRAP static Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
                                   int firstLevel=0, int WTA_K=2, ORB::ScoreType scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);
```
threshold：进行FAST检测时用到的阈值，**阈值越大检测到的角点越少**
###  8.5.3 流程
 1. 实例化FAST或OBR对象
 2. **将输入图像转灰度图**
 3. 根据需要，调用detect函数或compute函数或detectAndCompute函数，检测关键点和计算描述子
 4. 调用drawKeypoints函数绘制关键点
###  8.5.4 效果

```cpp
        Mat xuenai = imread("xuenai.jpg");
        imshow("xuenai", xuenai);
        namedWindow("panel");
        createTrackbar("threshold","panel", nullptr,255);
        createTrackbar("angle","panel", nullptr,360);
        createTrackbar("width","panel", nullptr,1000);
        createTrackbar("height","panel", nullptr,1000);

        while (1) {
            int thres = getTrackbarPos("threshold", "panel");
            if(thres==0)thres=100;
            int width = getTrackbarPos("width", "panel");
            if(width==0)width=xuenai.cols;
            int height = getTrackbarPos("height", "panel");
            if(height==0)height=xuenai.rows;
            int angle = getTrackbarPos("angle","panel");

            Mat xuenai_transform=xuenai.clone();

            resize(xuenai_transform,xuenai_transform,Size(width,height));

            Mat M= getRotationMatrix2D(Point2f(xuenai_transform.cols/2,xuenai_transform.rows/2),angle,1);
            warpAffine(xuenai_transform,xuenai_transform,M,xuenai_transform.size());

            Mat xuenai_gray(xuenai.size(),xuenai.type());
            cvtColor(xuenai_transform,xuenai_gray,COLOR_BGR2GRAY);

            Ptr<FastFeatureDetector>fast=FastFeatureDetector::create(thres);
            Ptr<ORB>obr=ORB::create();
            vector<KeyPoint>xuenai_FastKp,xuenai_ObrKp;
            fast->detect(xuenai_gray,xuenai_FastKp);
            obr->detect(xuenai_gray,xuenai_ObrKp);
            Mat fast_result=xuenai_transform.clone(),obr_result=xuenai_transform.clone();
            drawKeypoints(fast_result,xuenai_FastKp,fast_result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            drawKeypoints(obr_result,xuenai_ObrKp,obr_result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imshow("fast_result",fast_result);
            imshow("obr_result",obr_result);
            if (waitKey(0) == 'q')break;
        }
```
####  调整threshold
![在这里插入图片描述](https://img-blog.csdnimg.cn/33a3c6c4503e4e4c9775f3d0e12472c7.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e9465867a34e40dab8064f8faa2f14bf.png#pic_center)
####  进行缩放和旋转
![在这里插入图片描述](https://img-blog.csdnimg.cn/fd5bd4e1ff644d47847c2f5f04709853.png#pic_center)
####  错误

```cpp
fast->detectAndCompute(xuenai_gray,noArray(),fast_kp,fast_des);
```
前文已经提及，FAST算法不支持描述子的计算

```cpp
error: (-213:The function/feature is not implemented)  in function 'detectAndCompute'
```

##  8.6 Brute-Force与FLANN特征匹配
###  8.6.1 概述
####  Brute-Force
暴力匹配（Brute-force matcher）是最简单的二维特征点匹配方法。对于从两幅图像中提取的两个特征描述符集合，对第一个集合中的每个描述符Ri，从第二个集合中找出与其距离最小的描述符Sj作为匹配点。
暴力匹配显然会导致大量错误的匹配结果，还会出现一配多的情况。通过交叉匹配或设置比较阈值筛选匹配结果的方法可以改进暴力匹配的质量。

 - 如果参考图像中的描述符Ri与检测图像中的描述符Sj的互为最佳匹配，则称(Ri , Sj)为一致配对。交叉匹配通过删除非一致配对来筛选匹配结果，可以避免出现一配多的错误。
 - 比较阈值筛选是指对于参考图像的描述符Ri，从检测图像中找到距离最小的描述符Sj1和距离次小的描述符Sj2。设置比较阈值t∈[0.5 , 0.9]，只有当最优匹配距离与次优匹配距离满足阈值条件d (Ri , Sj1) ⁄ d (Ri , Sj2) < t时，表明匹配描述符Sj1具有显著性，才接受匹配结果(Ri , Sj1)。
####  FLANN
 - 相比于Brute-Force，FLANN的**速度更快**
 - 由于使用的是邻近近似值，所以**精度较差**

###  8.6.2 API
####  构造函数
```cpp
    CV_WRAP static Ptr<BFMatcher> BFMatcher::create( int normType=NORM_L2, bool crossCheck=false ) ;
    CV_WRAP static Ptr<FlannBasedMatcher> create();
    enum NormTypes {
                 NORM_INF       = 1,
                 NORM_L1        = 2,
                 NORM_L2        = 4,
                 NORM_L2SQR     = 5,
                 NORM_HAMMING   = 6,
                 NORM_HAMMING2  = 7,
                 NORM_TYPE_MASK = 7, 
                 NORM_RELATIVE  = 8,
                 NORM_MINMAX    = 32 
               };
```
normType：计算距离用到的方法，默认是欧氏距离。

crossCheck：是否使用交叉验证，默认不使用。

NORM_L1：L1范数，曼哈顿距离

NORM_L2：L2范数，欧氏距离

- **NORM_L1、NORM_L2适用于SIFT和SURF检测算法**

NORM_HAMMING：汉明距离

NORM_HAMMING2：汉明距离2，对每2个比特相加处理。

- **NORM_HAMMING、NORM_HAMMING2适用于OBR算法**
####  描述子匹配
```cpp
    CV_WRAP void DescriptorMatcher::match( InputArray queryDescriptors, InputArray trainDescriptors,
                CV_OUT std::vector<DMatch>& matches, InputArray mask=noArray() ) const;
```
queryDescriptors：描述子的查询点集，即参考图像的特征描述符的集合。

trainDescriptors：描述子的训练点集，即检测图像的特征描述符的集合。

- 特别注意和区分哪个是查询集，哪个是训练集

matches：std::vector\<DMatch\>匹配结果，长度为成功匹配的数量。

mask：掩码图像。**其大小与输入图像必须相同，且必须为灰度图**。计算时，对于掩码中的非0像素算法起作用，掩码中的灰度值为0的像素位置，算法不起作用。


```cpp
    CV_WRAP void DescriptorMatcher::knnMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   InputArray mask=noArray(), bool compactResult=false ) const;
```
queryDescriptors：描述子的查询点集，即参考图像的特征描述符的集合。

trainDescriptors：描述子的训练点集，即检测图像的特征描述符的集合。

- 特别注意和区分哪个是查询集，哪个是训练集


matches：std::vector<std::vector\<DMatch\>>类型，**对每个特征点返回k个最优的匹配结果**

k：返回匹配点的数量

mask：掩码图像。**其大小与输入图像必须相同，且必须为灰度图**。计算时，对于掩码中的非0像素算法起作用，掩码中的灰度值为0的像素位置，算法不起作用。
####  Brute-Force与FLANN对输入描述子的要求

 - **Brute-Force要求输入的描述子必须是CV_8U或者CV_32S**
 - **FLANN要求输入的描述子必须是CV_32F**

####  drawMatches绘制匹配结果
```cpp
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );
                             
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const int matchesThickness, const Scalar& matchColor=Scalar::all(-1),
                             const Scalar& singlePointColor=Scalar::all(-1), const std::vector<char>& matchesMask=std::vector<char>(),
                             DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );
                             
CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT );
```
img1(image1)：源图像1

keypoints1：源图像1的关键点

img2(image2)：源图像2

keypoints2：源图像2的关键点

matches1to2：源图像1的描述子匹配源图像2的描述子的匹配结果

outImg(out image)：输出图像

matchColor：匹配的颜色（特征点和连线)，默认Scalar::all(-1)，颜色随机

singlePointColor：单个点的颜色，即未配对的特征点，默认Scalar::all(-1)，颜色随机

matchesMask：掩码，决定哪些点将被画出，若为空，则画出所有匹配点

flags：特征点的绘制模式，其实就是设置特征点的那些信息需要绘制，那些不需要绘制。

###  8.6.3 流程
 1. 实例化BFMatcher对象
 2. 根据需要，调用match函数或knnMatch函数，进行特征匹配
 3. 调用drawMatches函数呈现原图，并且绘制匹配点
###  8.6.4 效果

```cpp
        Mat xuenai = imread("xuenai.jpg");
        Mat xuenai_rect = imread("xuenai_rect.jpg");
        cvtColor(xuenai_rect,xuenai_rect,COLOR_BGR2GRAY);
        imshow("xuenai", xuenai);
        namedWindow("panel");
        createTrackbar("threshold","panel", nullptr,255);
        createTrackbar("angle","panel", nullptr,360);
        createTrackbar("width","panel", nullptr,1000);
        createTrackbar("height","panel", nullptr,1000);

        while (true) {
            int thres = getTrackbarPos("threshold", "panel");
            if(thres==0)thres=100;
            int width = getTrackbarPos("width", "panel");
            if(width==0)width=xuenai.cols;
            int height = getTrackbarPos("height", "panel");
            if(height==0)height=xuenai.rows;
            int angle = getTrackbarPos("angle","panel");

            Mat xuenai_transform=xuenai.clone();

            resize(xuenai_transform,xuenai_transform,Size(width,height));

            Mat M= getRotationMatrix2D(Point2f((float )xuenai_transform.cols/2,(float )xuenai_transform.rows/2),angle,1);
            warpAffine(xuenai_transform,xuenai_transform,M,xuenai_transform.size());

            Mat xuenai_gray(xuenai.size(),xuenai.type());
            cvtColor(xuenai_transform,xuenai_gray,COLOR_BGR2GRAY);

            //准备工作
            Ptr<ORB>obr=ORB::create();
            vector<KeyPoint>xuenai_ObrKp;
            vector<KeyPoint>xuenai_rect_ObrKp;
            Mat BFMmatch_result;Mat FLANNmatch_result;
            Mat xuenai_obr_descriptorsForBF;Mat xuenai_rect_obr_descriptorsForBF;Mat xuenai_obr_descriptorsForFLANN;Mat xuenai_rect_obr_descriptorsForFLANN;
            vector<vector<DMatch>>xuenai_BFMmatch_results;vector<vector<DMatch>>xuenai_FLANNmatch_results;
            obr->detectAndCompute(xuenai_gray,noArray(),xuenai_ObrKp,xuenai_obr_descriptorsForBF);
            obr->detectAndCompute(xuenai_rect,noArray(),xuenai_rect_ObrKp,xuenai_rect_obr_descriptorsForBF);
            xuenai_obr_descriptorsForBF.convertTo(xuenai_obr_descriptorsForFLANN,CV_32F);
            xuenai_rect_obr_descriptorsForBF.convertTo(xuenai_rect_obr_descriptorsForFLANN,CV_32F);

            //进行匹配
            Ptr<BFMatcher>bfm=BFMatcher::create(NORM_HAMMING);
            Ptr<FlannBasedMatcher>flann=FlannBasedMatcher::create();
            bfm->knnMatch(xuenai_obr_descriptorsForBF,xuenai_rect_obr_descriptorsForBF,xuenai_BFMmatch_results,2);
            flann->knnMatch(xuenai_obr_descriptorsForFLANN,xuenai_rect_obr_descriptorsForFLANN,xuenai_FLANNmatch_results,2);

            //比率检验
            vector<DMatch>goodBFMresult,goodFLANNresult;
            for(auto match_result:xuenai_BFMmatch_results){
                if(match_result.size()>1 && match_result[0].distance<0.7*match_result[1].distance){
                    goodBFMresult.push_back(match_result[0]);
                }
            }
            for(auto match_result:xuenai_FLANNmatch_results){
                if(match_result.size()>1 && match_result[0].distance<0.7*match_result[1].distance){
                    goodFLANNresult.push_back(match_result[0]);
                }
            }
            
            //绘制匹配结果
            if(!goodBFMresult.empty()) {
                drawMatches(xuenai_rect, xuenai_rect_ObrKp,
                            xuenai_transform, xuenai_ObrKp,
                            goodBFMresult, BFMmatch_result,
                            Scalar::all(-1), Scalar::all(-1), vector<char>(),
                            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                imshow("BFMmatch_result",BFMmatch_result);
            }
            if(!goodFLANNresult.empty()) {
                drawMatches(xuenai_rect, xuenai_rect_ObrKp,
                            xuenai_transform, xuenai_ObrKp,
                            goodFLANNresult, FLANNmatch_result,
                            Scalar::all(-1), Scalar::all(-1), vector<char>(),
                            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                imshow("FLANNmatch_result",FLANNmatch_result);
            }
            if (waitKey(0) == 'q')break;
        }
```
####  不进行比率筛选
![在这里插入图片描述](https://img-blog.csdnimg.cn/58d3f5d59453442bb0d9f7c4240c2ac4.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5564da11f774c80877d3eba5ecf0c15.png#pic_center)

####  进行比率筛选
![在这里插入图片描述](https://img-blog.csdnimg.cn/5c15f17039d247ffa688f528003a3b73.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2fcaee0c53964ce0bb122e55bd905c5b.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8b9c699448564f539719ae8747125899.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2cb7a1c4290545b2be79eaf026b9b11e.png#pic_center)
#  9.结语与感谢
对于所有看到这儿的朋友们，我很高兴地祝贺你们已经完成了OpenCV的基础知识学习，我可以很明确地说，你们对计算机视觉领域的了解已经超过了绝大多数的人，恭喜你们。本文总计七万三千二百五十六字，三千四百一十二行，是本人第一次撰写的长篇系列教程，多有不当之处，谢谢读者们对我的包容和支持。
然而，纸上得来终觉浅，绝知此事要躬行。本文只是基础知识的讲解，关于OpenCV的实战项目，敬请关注本人的后续文章。
以上。
