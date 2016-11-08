#Ubuntu14.04上Caffe安装指南
##安装的准备工作
首先，安装官方版Caffe时，如果要使用Cuda，需要确认自己确实有NVIDIA GPU。
安装Ubuntu时，将/boot 分区分大概200M左右，太小了会导致升级系统时/boot空间不足。交换分区可以分到和机子的内存差不多。/opt 和 /usr/local 目录要保证空间能够满足软件安装的需求。临时目录也不能太小，建议10G以上，因为现在的Matlab、MKL软件都很大，临时目录可能挂载不上去。其余的差不多都可以分到/home了。
##开始安装
###更新系统
请注意，尽量不要更换Ubuntu的源，现在的官方源已经很快了。非官方源容易导致系统各库版本不兼容，让部分软件无法安装。

```
sudo apt-get update
sudo apt-get install build-essential
```
###安装CUDA

 1.  在[NVIDIA官网](https://developer.nvidia.com/cuda-downloads)下载CUDA，然后把名字改成一个简单点的，比如cuda.run
 2. `sudo chmod +x ./cuda.run`
 3. sudo service lightdm stop
 4. 经过第三步后会进入tty1命令行界面，输入自己的账号和密码。登录成功后，先cd到cuda下载的目录，输入`sudo ./cuda.run`。里面会有很多选项，目录就用默认的，选择Yes/No的时候就选Yes。
 5. 完成后我们还是回到图形界面吧，`sudo service lightdm start`
 6. `sudo gedit /etc/profile`，在文件末尾添加如下内容：```PATH=/usr/local/cuda-6.5/bin:$PATH
export PATH```
```
source /etc/profile
sudo gedit /etc/ld.so.conf.d/cuda.conf
```
这个文件是空的，在编辑器中输入：

```
/usr/local/cuda-7.5/lib64
```

在命令行输入：`sudo ldconfig`
至此，显卡驱动和cuda就安装好了，接下来安装cuda samples

```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev
```

cd到cuda samples的安装目录`cd /usr/local/cuda-7.5/samples`。

```
sudo make
```
完成后，

```
cd samples/bin/x86_64/linux/release
```

然后输入：`sudo ./deviceQuery`，如果能够打印出一系列的显卡信息，那么恭喜你，Cuda工作正常。
###安装数学库
如果你能够下载到MKL，并且有序列号，可以将MKL解压出来，然后cd到解压后的目录，`sudo ./install_GUI.sh`，里面的目录就默认的就行了，一路往下next，安装结束后`sudo gedit /etc/ld.so.conf.d/intel_mkl.conf`，在里面输入`/opt/intel/lib
/opt/intel/mkl/lib/intel64`，然后 `sudo ldconfig`更新一下库。至此就结束这一部分了。

如果你没有购买到MKL，那么可以使用atlas

```
sudo apt-get install libatlas-base-dev
```

###安装boost
在官网上下载boost源码，解压出来。cd到boost目录里。

```
bash ./bootstrap.sh
sudo ./b2 install
```
上面的代码可能会运行10-20分钟，你可以去喝杯咖啡了。
如果上面的代码没有报错，就可以运行`sudo ldconfig`了。

###安装opencv3.1

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev  libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
```

去[官网](https://codeload.github.com/Itseez/opencv/zip/3.1.0)下载一个opencv。然后解压出来。
在命令行中cd 到 opencv 里面后，一条条的运行下面的命令

```
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
```

上面的代码请一行行的运行，每一步都要确保没有报错。

###安装Python环境
虽然系统默认有python，但是我们需要用到Python的头文件等，必须安装python-dev，下面会安装这个，但是如果你嫌弃它安装的python版本太低（其实没必要嫌弃），你可以自己去官网上下载个最新的python2，然后编译。

```
sudo apt-get install python-dev python-pip
```

下载caffe的源码，然后解压出来。cd 到caffe-master/python里面，

```
for req in $(cat requirements.txt); do sudo pip install $req; done
```

注意，这一步可能会有些库安装失败，那你就需要自己去[PyPI](https://pypi.python.org/pypi)下载对应库，然后自己安装它。具体的库名称就在那个 requirements.txt里，后面的数字是版本号，可以不用管。
如果你自己喜欢Anaconda，也可以参考其他教程安装Anaconda。
###安装Matlab
如果你不用matlab或者不用Caffe的matlab借口（估计绝大部分人都不会去用），可以跳过这一步，真的！如果你要Matlab，

```
sudo mkdir /media/matlab
mount -o loop [path][filename].iso /media/matlab
cd /media/matlab
sudo ./install
```

安装过程中使用 readme.txt中的序列号。安装后使用crack中的license进行激活。
下面的路径是破解文件的路径，

```
sudo cp /路径/libmwservices.so /usr/local/MATLAB/R2014A/bin/glnxa64
```

```
sudo gedit /usr/share/applications/Matlab.desktop
```

输入：

```
[Desktop Entry]
Type=Application
Name=Matlab
GenericName=Matlab 2014a
Comment=Matlab:The Language of Technical Computing
Exec=sh /usr/local/MATLAB/R2014a/bin/matlab -desktop
Icon=/usr/local/MATLAB/Matlab.png
Terminal=false
Categories=Development;Matlab;
```

里面的东西根据自己的版本修改下。
###编译Caffe
先 cd 到你的 caffe-master，

```
cp Makefile.config.example Makefile.config
```

打开Makefile.config，看看里面的说明，根据自己的进行下配置。如果你一切都是按照默认的路径配置的，那就好办了。
如果你没有N卡，或者要使用CPU模式，那就把CUP_ONLY打开。
如果你使用的Opencv是3.1，就把 OPENCV_VERSION := 3前面的#去掉。
如果你用的MKL，就在BLAS := 后填入mkl。
其他的就按照自己的配置来吧。一般可以默认。

```
mkdir builds
cd builds
cmake ..
make all -j4
```

如果没有任何错误，那恭喜你，Caffe安装成功。
下面测试一下

```
make test
make runtest
```

test如果有几个错误或者FAIL，也算正常，不用太担心，错误可能是MKL的计算精度导致的。
接下来编译pycaffe

```
make pycaffe
```

编译 matcaffe

```
make matcaffe
```
