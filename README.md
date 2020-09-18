# SegLink

Detecting Oriented Text in Natural Images by Linking Segments (https://arxiv.org/abs/1703.06520).

## Prerequisites

The project is written in Python3 and C++ and relies on TensorFlow v1.0 or newer. We have only tested it on Ubuntu 14.04. If you are using other Linux versions, we suggest using Docker. CMake (version >= 2.8) is required to compile the C++ code. Install TensorFlow (GPU-enabled) by following the instructions on https://www.tensorflow.org/install/. The project requires no other Python packages.

On Ubuntu 14.04, install the required packages by
```
sudo apt-get install cmake
sudo pip install --upgrade tensorflow-gpu
```

## Supplement

There are some errors in the original code. For example, the flag 'pretrained_model' in original exp/sgd/pretrain.json isn't used in source code. I guess that the flag 'pretrained_model' should be 'vgg16_model'. I have fixed those problems which occur on following environment.

I have tested it on following environment:
```
Ubuntu 18.04 with CUDA8.0 & cuDNN5.1 & TensorFlow v1.0 & Anaconda3 Python3.5
```
It works well.

#### Usage

Firstly, please refer to Anaconda3 document for installing conda environment.

Then, while pip source and conda source don't have tensorflow-gpu v1.0 package, for installing TensorFlow-gpu you might need to run:
```
conda create -n seglink python=3.5
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp35-cp35m-linux_x86_64.whl
```

Then you can download VGG model for SSD by https://github.com/conner99/VGGNet link.

Then you should uncomment seglink/model_cnn.py:13 for converting caffe model to TensorFlow model.

**NOTE**: After converting caffe model to TensorFlow model, you should comment seglink/model_cnn.py:13, because it has been declared in seglink/solver.py.

Finally, you should see ``tool/create_datasets.py`` to generate TFRecord for training data and testing data.Training SegLink by running:
```
# example for pretraining
python ./manage.py train ./exp/sgd pretrain
```

## Installation

The project uses `manage.py` to execute commands for compiling code and running training and testing programs. For installation, execute
```
./manage.py build_op
```
in the project directory to compile the custom TensorFlow operators written in C++. To remove the compiled binaries, execute
```
./manage.py clean_op
```

## Dataset Preparation

See ``tool/create_datasets.py''

## Training

```
./manage.py <exp-directory> train
```

## Evaluation

See ``evaluate.py''
