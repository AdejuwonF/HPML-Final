Code for my HPML Final Project "Optimizing Deep Learning Inference on a MacBook Pro".  This repository contains an E2E process for training a model on the FER2013 dataset and then applying model
compilation, structured pruning and integer quantization to the model to imrpove performance.  It also contains code to benchmark performance and accuracy for these models.

## Prerequisites
- Download the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and place it in the same directory as these scrtipts
- Create a conda environment using environment.yml

## Running the Project E2E
The scripts should be run in the following order to recreate results

- Run through the [train_base_model.ipynb](https://github.com/AdejuwonF/HPML-Final/blob/master/train_base_model.ipynb) notebook to create the baseline model to optimize.  The model is defined in [common_utils.py](https://github.com/AdejuwonF/HPML-Final/blob/master/common_utils.py) so architecture can be changed there if you'd like.
This will create a folder of checkpoints of the model after every epoch.  These models will be saved as state_dicts by default and should be loaded as such.  You should analyze the training curves in Weights & Biases and decide on which checkpoint you'd like to use going forward.
- Next run through the [prune_model.ipynb](https://github.com/AdejuwonF/HPML-Final/blob/master/prune_model.ipynb) notebook.  This notebook will perform a sensitivity study on yourmodel (change the path to the model checkpoint you choose!!!).
After analyzing the sensitivity study you can choose an acceptable accuracy threshold and each layer in the model will be pruned accordingly.  The model will then be retrained with checkpoints being stored in another directory.  These checkpoints 
will be serialized as the complete model, not just the state dict, since we don't know beforehand what the model architecture will look like.  Analyze the training curves and choose which checkpoint you'd like to use going forward.
- Run the [quantize_models.ipynb](https://github.com/AdejuwonF/HPML-Final/blob/master/quantize_models.ipynb) notebook.  This will perform model quantization on both your chosen baseline model and the pruned model from the previous steps
(again update your paths accordingly).  By default these models will be saved as Pytorch JIT traces, since the quantized models can't be saved natively.
- Run the [benchmark_models.ipynb](https://github.com/AdejuwonF/HPML-Final/blob/master/benchmark_models.ipynb) notebook with the paths updated to your chosen baseline and pruned model.  This model benchmarks both performance and accuracy of the compiled
uncompiled version of your 4 models (baseline, quantized, pruned, pruned+ quantized).  It wil save these stats to Weights & Biases as well as produce matplot graphs comparing them.  The notebook will also profile the models using Pytorch's builtin profiler.
Results of the benchmarks will be saved as a pickled dictionary and the profile results for cpu usage and memory usage will be written to a text document.
- Finally edit [live_demo.py](https://github.com/AdejuwonF/HPML-Final/blob/master/live_demo.py) to load in your chosen model and run the python script.  It uses opencv to capture video from your webcam, so a compatible webcam is required.
