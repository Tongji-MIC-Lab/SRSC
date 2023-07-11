# Self-supervised Video Representation Learning by Serial Restoration with Elastic Complexity

Ziyu Chen, Hanli Wang, Chang Wen Chen


### Overview

Self-supervised video representation learning leaves out heavy manual annotation by automatically excavating supervisory signals. Although contrastive learning based approaches exhibit superior performances, pretext task based approaches still deserve further study. This is because the pretext tasks exploit the nature of data and encourage feature extractors to learn spatiotemporal logic by discovering dependencies among video clips or cubes, without manual engineering on data augmentations or manual construction of contrastive pairs. To utilize chronological property more effectively and efficiently, this work proposes a novel pretext task, named serial restoration of shuffled clips (SRSC), disentangled by an elaborately designed task network composed of an order-aware encoder and a serial restoration decoder. In contrast to other order based pretext tasks that formulate clip order recognition as a one-step classification problem, the proposed SRSC task restores shuffled clips into the right order in multiple steps. Owing to the excellent elasticity of SRSC, a novel taxonomy of curriculum learning is further proposed to equip SRSC with different pre-training strategies. According to the factors that affect the complexity of solving the SRSC task, the proposed curriculum learning strategies can be categorized into task based, model based and data based. Extensive experiments are conducted on the subdivided strategies to explore their effectiveness and noteworthy laws. Compared with existing approaches, this work demonstrates that the proposed approach achieves state-of-the-art performances in pretext task based self-supervised video representation learning and a majority of the proposed strategies further boost the performance of downstream tasks. For the first time, the features pre-trained by the pretext tasks are applied to video captioning by feature-level early fusion, and enhance the input of existing approaches as a lightweight plugin.


### Method

The pipeline of the proposed task network OAE-SRD for solving the proposed SRSC task is shown in Fig. 1. (a) Feature Extraction: n sampled clips are first shuffled and then fed into 3D backbone CNNs to obtain raw features. (b) Order-Aware Encoder: The raw features are fed into transformer without positional encoding to obtain order-aware features. Global aggregation is utilized to obtain a compact representation as the initial input for decoder. (c) Serial Restoration Decoder: at each step, the hard attention pointer (HAP) outputs the probability of each clip being restored, and the most probable one is chosen and fed as the next input of LSTM. The optional mask is used to adjust task complexity for model based curriculum learning.

<p align="center">
<image src="source/fig1.png" width="700">
<br/><font>Fig. 1. Overview of the proposed SRSC task and model architecture.</font>
</p>


### Results

The proposed SRSC task with OAE-SRD task network is compared with several state-of-the-art self-supervised video representation learning methods in Table 1. For a comprehensive comparison, we have summarized the techniques for generating supervisory signals, i.e., construction of positive and negative samples, data augmentation, pseudo-labeling, and utilization of intrinsic properties. These techniques are abbreviated as Pos-Neg, DA, PL and Prop in the header of Table I. As each contrastive learning based SSL has its unique Pos-Neg, DA or PL, we use the checkmark (✓) simply to indicate whether some of the three techniques are used. For intrinsic properties (Prop), we have summarized seven types, i.e., CO for chronological order, STL for space-time layout, GI for geometric information, SI for statistical information, PR for playback rate (also called speed or sampling rate in some methods), PE for periodicity (also called palindrome in some methods) and VD for video continuity in Table I. In the "Type" column, "C" represents contrastive learning based approaches, "P" represents pretext task based approaches and dagger "†" means that the modified approaches work as a plugin. In Table 2, the dashed lines divide the table into three parts, which from top to bottom exhibit the results of task based, model based and data based curriculum learning strategies in terms of action recognition task. In Table 3, the proposed SRSC task is compared with other state-of-the-art self-supervised video representation learning methods in terms of nearest neighbor retrieval task. Table 3 also exhibits the performance of nearest neighbor retrieval with the proposed curriculum learning strategies. Table 4 exhibits the performance of existing video caption methods fused with the features pre-trained by SRSC task.

<p align="center">
<font>Table 1. Comparison between the proposed SRSC task and state-of-the-art self-supervised video representation learning methods in terms of action recognition performance on UCF101 and HMDB51.</font><br/>
<image src="source/table1.png" width="600">
</p>

<p align="center">
<font>Table 2. Results of the proposed curriculum learning strategies in terms of action recognition performance on UCF101 and HMDB51.</font><br/>
<image src="source/table2.png" width="320">
</p>

<p align="center">
<font>Table 3. Results of the proposed curriculum learning strategies in terms of nearest neighbor retrieval performance on UCF101 and HMDB51.</font><br/>
<image src="source/table3.png" width="500">
</p>

<p align="center">
<font>Table 4. Results for video captioning methods fused with the features of the SRSC task.</font><br/>
<image src="source/table4.png" width="550">
</p>


### Environments

* System Level
  * Ubuntu 18.04.5
  * NVIDIA Driver Version: 455.23.05
  * gcc 7.5.0
  * ffmpeg 3.4.8
  * GPU: Tesla V100 32GB (SRSC tasks occupy 20-30GB)
* Conda Environment Level
  * Python 3.8.5
  * Pytorch 1.7.1
  * cudatoolkit 11.0
  * cudnn 8.0.5
  * requirements.txt (which contains main dependency packages)


### Directory Structure

    ├── README.md
    ├── data
    ├── datasets
    │   ├── __init__.py
    │   ├── hmdb51.py
    │   ├── kinetics400.py
    │   └── ucf101.py
    ├── requirements.txt
    ├── scripts_srsc
    │   ├── ft_classify.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── c3d.py
    │   │   ├── r21d.py
    │   │   ├── r3d.py
    │   │   ├── srsc.py
    │   │   ├── srsc_ablation.py
    │   │   ├── task_network.py
    │   │   ├── task_network_ablation.py
    │   │   └── task_network_mask.py
    │   ├── profile_srsc.py
    │   ├── retrieve_clips.py
    │   ├── train_classify.py
    │   ├── train_srsc.py
    │   ├── train_srsc_ablation.py
    │   ├── train_srsc_kin.py
    │   └── utils.py
    └── video_caption
        ├── RMN_R21D
        │   ├── evaluate.py
        │   ├── models
        │   │   ├── RMN.py
        │   │   ├── __init__.py
        │   │   ├── allennlp_beamsearch.py
        │   │   ├── framework.png
        │   │   └── gumbel.py
        │   ├── train.py
        │   └── utils
        │       ├── Vid2Url_Full.txt
        │       ├── data.py
        │       ├── opt.py
        │       └── utils.py
        ├── RMN_R3D_C3D
        │   ├── evaluate.py
        │   ├── models
        │   │   ├── RMN.py
        │   │   ├── __init__.py
        │   │   ├── allennlp_beamsearch.py
        │   │   ├── framework.png
        │   │   └── gumbel.py
        │   ├── train.py
        │   └── utils
        │       ├── Vid2Url_Full.txt
        │       ├── data.py
        │       ├── opt.py
        │       └── utils.py
        ├── SAAT_R21D
        │   ├── Makefile_msrvtt_svo
        │   ├── Makefile_yt2t_svo
        │   ├── dataloader_svo.py
        │   ├── model_svo.py
        │   ├── opts_svo.py
        │   ├── test_svo.py
        │   └── train_svo.py
        ├── SAAT_R3D_C3D
        │   ├── Makefile_msrvtt_svo
        │   ├── Makefile_yt2t_svo
        │   ├── dataloader_svo.py
        │   ├── model_svo.py
        │   ├── opts_svo.py
        │   ├── test_svo.py
        │   └── train_svo.py
        ├── SGN_R21D
        │   ├── config.py
        │   ├── loader
        │   │   ├── MSRVTT.py
        │   │   ├── MSVD.py
        │   │   ├── __init__.py
        │   │   ├── data_loader.py
        │   │   └── transform.py
        │   ├── models
        │   │   ├── __init__.py
        │   │   ├── attention.py
        │   │   ├── decoder.py
        │   │   ├── densenet.py
        │   │   ├── pre_act_resnet.py
        │   │   ├── resnet.py
        │   │   ├── resnext.py
        │   │   ├── semantic_grouping_network.py
        │   │   ├── transformer
        │   │   │   ├── Constants.py
        │   │   │   ├── Layers.py
        │   │   │   ├── Models.py
        │   │   │   ├── Modules.py
        │   │   │   ├── SubLayers.py
        │   │   │   └── __init__.py
        │   │   ├── visual_encoder.py
        │   │   └── wide_resnet.py
        │   └── utils.py
        └── SGN_R3D_C3D
            ├── config.py
            ├── loader
            │   ├── MSRVTT.py
            │   ├── MSVD.py
            │   ├── __init__.py
            │   ├── data_loader.py
            │   └── transform.py
            ├── models
            │   ├── __init__.py
            │   ├── attention.py
            │   ├── decoder.py
            │   ├── semantic_grouping_network.py
            │   ├── transformer
            │   │   ├── Constants.py
            │   │   ├── Layers.py
            │   │   ├── Models.py
            │   │   ├── Modules.py
            │   │   ├── SubLayers.py
            │   │   └── __init__.py
            │   └── visual_encoder.py
            └── utils.py

- `datasets`: dataset, dataloader and preprocessing for *HMDB51* & *Kinetics400* & *UCF101*.

- `scripts_srsc`: base task training, relay task training, evaluation and testing

- `requirements.txt`: main package dependencies

- `data`: contains `hmdb51` ,`ucf101` and `compress` directory, each of which has a `split` directory for train/test/validation splits

- Video data also stored in `hmdb51` ,`ucf101` and `compress`, and the download links are as follows:
  
  - [*UCF101*](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
  - [*HMDB51*](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)
  - [*Kinetics400*](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz)

- `video_caption`: contains codes for additional experiments on video captioning. We reference codes of three recent works, which are [IJCAI2020:RMN](https://github.com/tgc1997/RMN), [CVPR2020:SAAT](https://github.com/SydCaption/SAAT) and [AAAI2021:SGN](https://github.com/hobincar/SGN). 


### Commands

- Install dependency packages:
  
  - `pip install requirements.txt`

- Get video lists which satisfy 'non-overlapped, **number of clips(tl)**, **clip length(cl)** and **interval(it)**' by running following commands:
  
  - `cd datasets`
  - `python hmdb51.py` or `python ucf101.py` or `python kinetics400.py`
  
  Remeber to change parameters (`tl`, `cl` and `it`) in method `gen_*_srsc_splits` in different experimental settings.

- Go into working directory and create soft links:
  
  - `cd ../scripts_srsc`
  - `ln -s ../datasets datasets`
  - `ln -s ../data data`

- Train base task by running following commands:
  
  - `python train_srsc.py --model $model_name --log $log_dir --workers $num_workers --gpus $gpu_id`
  - `--mask` is switched on if derived task network is used.

- Train relay task with **task based** or **model based** strategies by running following command:
  
  - `python train_srsc.py --model $model_name --log $log_dir --ckpt $log_dir/$pre_task_log_dir/$best_model --gpus $gpu_id --workers $num_workers --tl $number_of_clips`
  - `--mask` is switched on if derived task network is used.

- Train relay task with **data based** strategy by running following command:
  
  - `python train_srsc_kin.py --log $log_dir --model $model_name --ckpt $log_dir/$pre_task_log_dir/$best_model --gpus $gpu_id --workers $num_workers  --epochs $num_epochs`

- Ablation study on OAE or SRD by running following command:
  
  - `python train_srsc_ablation.py --model $model_name --log $log_dir --workers $num_workers --gpus $gpu_id --ablation OAE`
  
  - `python train_srsc_ablation.py --model $model_name --log log_dir --workers $num_workers --gpus $gpu_id --ablation SRD`

- Finetune on action recognition task by running following command:
  
  - `python ft_classify.py --model $model_name --log $log_dir --ckpt $log_dir/$task_log_dir/$best_model --gpus $gpu_id --workers $num_workers --split $split`

- Evaluate on action recognition task by running following command:
  
  - `python train_classify.py --mode $mode --model $model_name --ckpt $log_dir/$finetune_log_dir/$best_model --gpus $gpu_id --workers $num_workers --split $split`

- Evaluate on nearest neighbor retrieval task by running following command:
  
  - `python retrieve_clips.py --model $model_name --feature_dir data/features/$dataset_name/$model_name --workers $num_workers --ckpt $log_dir/$task_log_dir/$best_model --gpu $gpu_id --dataset $dataset_name`

- Evaluate on video captioning by running following commands:
  
  - Enter SRSC project root directory: `cd ..`
  
  - Clone RMN, SGN and SAAT: `git clone $PROJECT_URL`
  
  - Replace files or directories with same names in `video_caption` directory. *E.g.*, `video_caption/SGN_R21D/config.py` should replace the `config.py` in original codes from SGN repository. 
  
  - Follow the training and evaluation commands in original repositories. 

- Visualize training/finetuning with `tensorboard` by runninng following command:
  
  - `tensorboard --logdir $log_dir --port $port_number`

- **Note**
  
  - Most hyper-parameters have their default values set in scripts, and they can be re-set in *Commond Line* with regard to specific task.
  
  - Terminology Matching
    
    | In paper                                         | In code / CMD                          |
    | ------------------------------------------------ | -------------------------------------- |
    | number of clips / sequence length                | `tl` / `$number_of_clips`              |
    | the derived one / the derived network            | `--mask`                               |
    | clips are spaced by  $ n $ frames                 | `it` / `interval`                      |
    | consecutive $ m $ frames are grouped into a clip | `cl` / `clip_length`                   |
    | backbone network / feature extractor             | `r21d` / `r3d` / `c3d` / `$model_name` |
  
  - As batch size is set to 8 (relatively small), the multi-gpus training is not recommended, which reduces samples in BatchNorm layers and makes estimation of variance and mean unstable. 

​           


### Acknowledgement

Thanks to [Xu's code](https://github.com/xudejing/video-clip-order-prediction), we referenced it and implemented our approach.  

For video captioning part, we also referenced [SGN](https://github.com/hobincar/SGN), [RMN](https://github.com/tgc1997/RMN) and [SAAT](https://github.com/SydCaption/SAAT).

Feel free to contact us via ziyuchen1997@tongji.edu.cn or open an issue on github if there is any problem.


### Citation

Please cite the following paper if you find this work useful:

Ziyu Chen, Hanli Wang, Chang Wen Chen, Self-supervised Video Representation Learning by Serial Restoration with Elastic Complexity, IEEE Transactions on Multimedia (TMM'23), accepted, 2023.
