# -*- coding: utf-8 -*-
# https://blog.csdn.net/tangjiahao10/article/details/125227005
# https://blog.csdn.net/qq_40839674/article/details/121826900  Paddle detection 笔记（自用）
# python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# cuDNN Version: 8.4.    18.0
# 如果GPU卡数或者batch size发生了改变，你需要按照公式 lrnew = lrdefault * (batch_sizenew * GPU_numbernew) / (batch_sizedefault * GPU_numberdefault) 调整学习率。
# 这是测试的一些命令语句：https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/docs/tutorials/QUICK_STARTED_cn.md
# 焕然一新的PaddleDetection  https://aistudio.baidu.com/aistudio/projectdetail/1859839?channelType=0&channel=0
# http://localhost:8040/app/scalar
# https://aistudio.baidu.com/aistudio/projectdetail/1246199 可视化Loss曲线

# pip uninstall mmcv           先卸载再安装
# pip uninstall mmcv-full
# mim install mmcv==1.6.0   需要1.6.0版本的mmcv 在h2rbox中
# mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
#  pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
# mim install mmdet==2.20.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install mmdet==2.20.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install setuptools==59.5.0

# nvidia-smi -L
# nvidia-smi
# nvcc -V
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple     清华源
# rm -r 文件夹

# pip uninstall mmcv-full

# pip install mmcv==1.4.5  尽量不要用pip安装mmcv
# nvidia-smi
# https://blog.csdn.net/qq_43885462/article/details/125668170     CUDA安装
# https://www.cnblogs.com/blue-lin/p/16375265.html   CUDA安装
# https://blog.csdn.net/weixin_45811857/article/details/124457280               CUDA安装
# nvcc -V
# ls -l /usr/local
#  conda activate h2rbox-mmrotate   激活虚拟环境

# uname -v 查看当前 Linux 系统的版本

# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/h2rbox/ssdd/h2rbox_r50_adamw_fpn_15x_ssdd_le90.py --work-dir work_dirs/ssdd/h2rbox_adamw

# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch    在QPDet中用的这个
# pip install -U openmim   在QPDet中用的这个
# mim install mmcv-full    在QPDet中用的这个

#   在QPDet中用的这个用mmcv==1.6.2
# mmdet/ops/conv_ws.py
# the 23th line:
# @CONV_LAYERS.register_module('ConvWS')
# modified:
# @CONV_LAYERS.register_module(name='ConvWS', force=True)
# the 52th line as above.

# the reason that mmcv edition may be higher.


# cd BboxToolkit/tools/configs/  
# vi ss_sar_train.json      如果不能编辑打开后按 i 键
# 返回先按ESC  然后冒号：再按 wq！按回车
# du -h -d 1


# ssh djn@192.168.50.14
# cd /mnt/data/docker/volumes/

# cd ws-guopengfei-mlhub-PaddleDetection-116-840
# du -h -d 1
# cd _data
# exit

# CUDA_VISIBLE_DEVICES=0 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_r16_coco_ssdd.yml --eval 


# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_1_r16_coco_ssdd.yml --eval -o use_gpu=true



# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_cheng_r16_coco_ssdd.yml --eval -o use_gpu=true
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_1_cheng_r16_coco_ssdd.yml --eval -o use_gpu=true



# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_cheng_r16_coco_ssdd.yml --eval -o use_gpu=true

# CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_cheng_r16_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar0
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_cheng_r16_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./output"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r16_2he_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./output0"
##########################################PaddleDetection/PaddleDetection/outputqrheqkv_ESE1
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_1_layerchannel_r16_qrheqkv_BB_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputqrheqkv_BB"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r148_qrheqkv_BB_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr148qrheqkv_BB"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_2iaarjian0dain5_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_2iaarjian0dain5"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_LcLb_2iaarjian0dain7_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_LcLb_2iaarjian0dain7"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_Lc_sigai_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_Lc_sigai"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_Lc_sigaijia0dian5_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_Lc_sigaijia0dian5"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_Lc_hanshuaijia0dian5_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_Lc_hanshuaijia0dian5"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_iou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_Lc_iou"





#########################################################

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/detr/detr_r50_1x_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputdetr_r50_1x_ssdd"









# PaddleDetection/configs/rtdetr/PaddleDetection/configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r148_qrheqkv_BB_coco_ssdd.yml
# python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_avg_cheng_r16_coco_ssdd.yml --eval



# CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch  tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_chengvdl_r16_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar


# CUDA_VISIBLE_DEVICES=1 python -m paddle.distributed.launch  tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_1_layerchannel_chengvdl_r32_coco_ssdd.yml --eval --use_vdl=true --vdl_log_dir=vdl_dir/scalar 



# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_coco_ssdd.yml --eval 

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_1x_coco_ssdd.yml --eval


# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_spine.yml --eval 



# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_ssdd.yml --eval
#########################################################################################################

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_iou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_Lc_iou"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r137_qrheqkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr137qrheqkv_BB_ciou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r135_qrheqkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr135qrheqkv_BB_ciou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r159_qrheqkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr159qrheqkv_BB_ciou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r157_qrheqkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr157qrheqkv_BB_ciou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_ciou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_SIiou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_SIiou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_zongarfSIiou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_zongarfSIiou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_iou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_iou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_giou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_giou" 
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1510_qrheqkv_BB_diou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1510qrheqkv_BB_diou"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_layerchannel_r1612_qrheqkv_BB_zongarfSIiou.yml --eval --use_vdl=true --vdl_log_dir="./outputr1612qrheqkv_BB_zongarfSIiou"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_L_6x_0_layerchannel_r1510_qrheqkv_BB_zongarfSIiou.yml --eval --use_vdl=true --vdl_log_dir="./outputr_L_1510qrheqkv_BB_zongarfSIiou"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_iou.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_iou"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_iou_hrsid.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_iou_hrsid"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_layerchannel_r1510_qrheqkv_BB_zongarfSIiou_hrsid.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_r1510_qrheqkv_BB_zongarfSIiou_hrsid"


# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_layerchannel_r1510_qrheqkv_BB_ciou_hrsid.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_r1510_qrheqkv_BB_ciou_hrsid"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_layerchannel_r137_qrheqkv_V_ciou_hrsid.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_r137_qrheqkv_V_ciou_hrsid"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM.yml"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM_MPE.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM_MPE.yml"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM_BB.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM_BB"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM_BB_ciou"


# python tools/infer.py -c configs/yolov3/yolov3_mobilenet_v3_large_ssld_270e_voc.yml \  
# -o weights=output/yolov3_mobilenet_v3_large_ssld_270e_voc/best_model.pdparams \
# --infer_img=/home/aistudio/work/fall4.jpg

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM_BB_inciou.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM_BB_inciou"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_ssdd_MRFM_BB_incioumy.yml --eval --use_vdl=true --vdl_log_dir="./outputr_x_layerchannel_ssdd_MRFM_BB_incioumy"

# python tools/infer.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_iou.yml \  
# -o weights=output/rtdetr_hgnetv2_x_6x_0_iou/model_final.pdparams \
# infer_img=demo/232.png


# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_l_6x_ssdd.yml --eval --use_vdl=true --vdl_log_dir="./outputr_l_rtdetr_hgnetv2_l_6x_ssdd"


# python -u tools/eval.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_iou.yml  -o weights=output/rtdetr_hgnetv2_x_6x_0_iou/best_model.pdparams


# ###############################################################################################################################
# PaddleDetection/ppdet/modeling/transformers/rtdetr_transformer.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_mrfm_qkv_BB_ciou.yml --eval --use_vdl=true --vdl_log_dir="./outputrtdetr_hgnetv2_x_6x_0_mrfm_qkv_BB_ciou_ssdd"


# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_hgnetv2_x_6x_0_qkv_BB_iou.yml --eval --use_vdl=true --vdl_log_dir="./outputrtdetr_hgnetv2_x_6x_0_qkv_BB_iou_ssdd"

