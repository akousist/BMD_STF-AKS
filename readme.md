# BMD
## About

## Installation
1. Install Python 3.10 on your computer
2. Install pipenv with `pip install pipenv`
3. Set working directory to this project's path
4. Run `pipenv install` to build up environment

## How to use
To predict BMD condition from a hand radiograph:
```
pipenv run predict path_to_radiography_image
```
e.g. `pipenv run predict test.jpg`
### Trouble Shooting
```
Model weights missing: resnet_weights.pt
```
Default weight path is `[project_directory]/resnet_weights.pt`. Check if the model weight is downloaded and place in the right directory.
```
Image does not exists: [some file path]
```
Image to be analyzed is not placed according to the path provided to the program.
### Additional parameter
If the host machine is cuda-available and the package `torch` is installed and compiled with cuda, one can run the code with GPU by:
```
pipenv run predict test.jpg --cuda
```
Default path to model weights can be changed with `--resnet_weights` flag:
```
pipenv run predict test.jpg --resnet_weights=another_weights.pt
```
