# EDT 

A PyTorch implementation of [**Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing**](https://doi.org/10.1145/3746252.3761408).



## Requirements

```
conda env create -f EDT.yml
```

## Run the code
The evaluation python files are located at ```evaluate/{dataset}/{model}.py```

## Example
For simplification, we showed a simple example at ```cifar10.ipynb```, which backdoored a ship image to a cat label. You can also refer to ```example.ipynb``` for some detailed explanation.

## Citation
If you find this repo useful, please consider citing:
```
@inproceedings{10.1145/3746252.3761408,
author = {Guo, Dongliang and Hu, Mengxuan and Guan, Zihan and Guo, Junfeng and Hartvigsen, Thomas and Li, Sheng},
title = {Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3761408},
doi = {10.1145/3746252.3761408},
booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
pages = {750–760},
numpages = {11},
keywords = {large pre-trained model, model editing, safety},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```
