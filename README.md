**Official code** for our paper: [Interpreting and Analysing CLIP's Zero-Shot Image Classification via Mutual Knowledge](https://arxiv.org/pdf/2410.13016), NeurIPS 2024

![demo](demo.png)

In order to keep this repo clean and readable, only the code for a single example is provided. If you need codes for other analysis/ablations/experiments/datasets, please submit an issue or email me, and i will provide it directly. 

To avoid running the CLIP text encoder on all classes and descriptors everytime, we save its output as the weights of a [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) layer for ease and convenient loading. Please download the imagenet classifiers from [here](https://drive.google.com/drive/folders/1d4nWRuCOvLxfCgzn4-XltXRWW6l0ghZJ) and place in `/classifiers` in this repo. Or you can just build them yourself (they are just running the text inputs on CLIP Text Encoder and saving the output as the weights of a linear layer in PyTorch). [This](https://drive.google.com/file/d/1PGNPXOhl-sDe-UhxEA0MmPB07LuVkcO5/view) is the code for this. 

### Additional Libraries

- [clip](https://github.com/openai/CLIP)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [kmeans pytorch](https://github.com/subhadarship/kmeans_pytorch)
- [SimpleCRF](https://github.com/HiLab-git/SimpleCRF)
- [matplotlib-venn](https://pypi.org/project/matplotlib-venn/)


