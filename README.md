# Adversarial-defense

## Dependencies
torch 1.12.1+cu113

advertorch 0.2.3

torchattacks 3.5.1

## Dataset Preparation
We use the dataset loaders for the Cifar-10. Put it in that folder, or change the default file location in my_test.py

## Models
We present our approach along with ResNet18 of AT in two my_model files. The detailed design of both models can be found in resnet.py.


## Usage
### testing our method 
To test our method, we need to go into the ours folder.
```
cd ~/ours/
```
Change the path to the cifar10 file and the path to model in my_test.py. Then just run my_test.py.
```
python my_test.py 
```

### testing compare method
To test compare method, we need to go into the AT folder.
```
cd ~/AT/
```
Change the path to the cifar10 file and the path to model in my_test.py. Then just run my_test.py.
```
python my_test.py 
```

## Result
We present the accuracy under natural samples, PGD,DDN,CW, and STA.
