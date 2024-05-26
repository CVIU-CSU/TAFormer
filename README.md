# Task-Aware Transformer For Partially Supervised Retinal Fundus Image Segmentation    

## Environment

This code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/)

* python == 3.8

* Pytorch == 1.9.0

* timm

* imgviz

```shell
conda create --name mmseg-v1rc python=3.8 -y
conda activate mmseg-v1rc

conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch -y

pip install -U openmim
mim install "mmengine==0.3.2"
mim install "mmcv==2.0.0rc3"

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout -b 1rc2 v1.0.0rc2
pip install -v -e . -i https://pypi.douban.com/simple/

mim install "mmdet==3.0.0rc4"
mim install "mmcls==1.0.0rc4"

pip install imgviz
pip install timm
pip install kornia==0.5.8
```

Note that replacing the appropriate packages with the given mmengine, mmcv and mmdet.

## Dataset Preparations

Please see [PSSNet/prepare_dataset](https://github.com/CVIU-CSU/PSSNet/tree/master/prepare_dataset)

## Training

```shell
bash mmsegmentation/tools/dist_train.sh mmsegmentation/configs/_mask2former_/mask2former_swin-t_4xb2-40k_multi_dataset-512x512_90query_6layer_mean_teacher_query_mask_random_shift_learn_generate_query_mask_feature_pseudo_factor.py 4
```

## Evaluation

```shell
python mmsegmentation/tools/vis.py images_path ann_path 7 config_file_path checkpoint_path
```

## Models

We provide the final model and training logs [here](https://pan.baidu.com/s/1OMHKG6uWn4-rizZ5Hywgqw?pwd=8fcz)

## Citation

If you find it useful for your your research and applications, please cite using this BibTeX:

```none
@inproceedings{zeng2024task,
  title={Task-Aware Transformer For Partially Supervised Retinal Fundus Image Segmentation},
  author={Zeng, Hailong and Liu, Jianfeng and Liang, Yixiong},
  booktitle={2024  International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
```

