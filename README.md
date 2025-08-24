# M2S-DETR

![image text](https://github.com/douling843/M2S-DETR/blob/main/fig1.jpg)  


⚡   This repository includes the official implementation of the paper:  

👋  **M2S-DETR: A Mixed Receptive Field and Multi-Position Encoding for Scale-Sensitive SAR Ship Detection with Transformer**

👨‍💻   **Code:** [GitHub](https://github.com/douling843/M2S-DETR/edit/main)



## Installation  <img src="fig2/Installation.svg" width="4%">
环境要求：
- <span> PaddlePaddle 2.3.2
- <span> OS 64位操作系统
- <span> Python 3(3.5.1+/3.6/3.7/3.8/3.9/3.10)，64位版本
- <span> pip/pip3(9.0.1+)，64位版本
- <span> CUDA >= 10.2
- <span> cuDNN >= 7.6


1.安装PaddlePaddle

 CUDA10.2
`python -m pip install paddlepaddle-gpu==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

 CPU
`python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"


2. 安装M2S-DETR
注意： pip安装方式只支持Python3


 克隆仓库
`cd <path/to/clone/M2S-DETR>`
`git clone https://github.com/douling843/M2S-DETR.git`

 安装其他依赖
`cd M2S-DETR`
`pip install -r requirements.txt`

 编译安装
`python setup.py install`


注意

如果github下载代码较慢，可尝试使用gitee或者代理加速。

若您使用的是Windows系统，由于原版cocoapi不支持Windows，pycocotools依赖可能安装失败，可采用第三方实现版本，该版本仅支持Python3

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

若您使用的是Python <= 3.6的版本，安装pycocotools可能会报错distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3'), 您可通过先安装cython如pip install cython解决该问题

安装后确认测试通过：

python ppdet/modeling/tests/test_architectures.py
测试通过后会提示如下信息：

.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK

快速体验
恭喜！ 您已经成功安装了，接下来快速体验目标检测效果

在GPU上预测一张图片
`export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg`
会在output文件夹下生成一个画有预测结果的同名图像。


## Acknowledgement  📫

This repository is based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1) 👯.
