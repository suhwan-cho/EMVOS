# EMVOS

This is the official PyTorch implementation of our paper:

> **Pixel-Level Equalized Matching for Video Object Segmentation**, *arXiv'22*\
> [Suhwan Cho](https://github.com/suhwan-cho), Woo Jin Kim, MyeongAh Cho, [Seunghoon Lee](https://github.com/iseunghoon), [Minhyeok Lee](https://github.com/Hydragon516), [Chaewon Park](https://github.com/codnjsqkr), Sangyoun Lee

URL: [Official] [[arXiv]](https://arxiv.org/abs/2209.03139)\
PDF: [Official] [[arXiv]](https://arxiv.org/pdf/2209.03139.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208472042-2e87b90f-65d9-4118-b526-99657bf36ccb.png" width=800>

```
@article{EMVOS,
  title={Pixel-Level Equalized Matching for Video Object Segmentation},
  author={Cho, Suhwan and Kim, Woo Jin and Cho, MyeongAh and Lee, Seunghoon and Lee, Minhyeok and Park, Chaewon and Lee, Sangyoun},
  journal={arXiv preprint arXiv:2209.03139},
  year={2022}
}
```
You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).



## Abstract
In surjective matching, the matching is performed flexibly as there are no restrictions on the matching process. This makes surjective matching susceptible to background distractions. To prevent this, bijective matching mechanisms have been introduced to capture the locality of a video, but they still suffer from two limitations: 1) surjective matching cannot be fully leveraged; and 2) test-time manual tuning is required for searching the optimal hyper-parameters. In this paper, we propose an **equalized matching mechanism** that does not suffer from the limitations of existing bijective matching mechanisms, while sustaining the same advantages.  By simply plugging an equalized matching branch to the baseline model, a significant performance boost can be obtained.


## Preparation
1\. (Optional) Download [COCO](https://cocodataset.org/#download) for network pre-training.

2\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) for network main training and testing.

3\. Download [YouTube-VOS](https://youtube-vos.org/dataset/vos) for network main training and testing.

4\. Replace dataset paths in *"run.py"* file with your dataset paths.



## Training
1\. Select training datasets in *"run.py"* file.

2\. Run **EMVOS** training!!
```
python run.py --train
```


## Testing
1\. Make sure the pre-trained models are in your *"trained_model"* folder.

2\. Select a pre-trained model and testing datasets in *"run.py"* file.

3\. Run **EMVOS** testing!!
```
python run.py --test
```

4\. (Optional) Download [pre-computed results](https://drive.google.com/file/d/1GWn7csEX4C8PJWcnC1ykS8IYobloQaSh/view?usp=sharing).



## Note
Code and models are only available for non-commercial research purposes.

If you have any questions, please feel free to contact me :)
```
E-mail: chosuhwan@yonsei.ac.kr
```
