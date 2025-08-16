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
