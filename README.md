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

# CUDA10.2
`python -m pip install paddlepaddle-gpu==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

# CPU
`python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple`

# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"


2. 安装M2S-DETR
注意： pip安装方式只支持Python3


# 克隆PaddleDetection仓库
`cd <path/to/clone/M2S-DETR>`
`git clone https://github.com/douling843/M2S-DETR.git`

# 安装其他依赖
`cd M2S-DETR`
`pip install -r requirements.txt`

# 编译安装
`python setup.py install`
