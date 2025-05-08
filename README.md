# CMPE-258-Assignment-6-Training-Neural-networks-with-Keras-Advanced

This repository contains a comprehensive exploration of advanced techniques for training neural networks using Keras and TensorFlow. The project is divided into two main parts focusing on regularization, data augmentation, and custom Keras components.

## Project Overview

### Part 1: Regularization & Data Augmentation

#### A. Regularization Techniques
This section demonstrates various regularization and generalization techniques with A/B testing to show their effectiveness:

- **L1L2_Regularization.ipynb**: Implementation and comparison of L1 and L2 regularization methods
- **Dropout_Regularization.ipynb**: Standard dropout implementation with performance analysis
- **Early_Stopping.ipynb**: Early stopping techniques to prevent overfitting
- **MonteCarlo_Dropout.ipynb**: Using dropout at inference time for uncertainty estimation
- **Weight_Initializations.ipynb**: Comparison of different weight initialization strategies
- **Batch_Normalization.ipynb**: Implementation of batch normalization and its effects
- **Custom_Regularization.ipynb**: Creating custom regularization techniques
- **Callbacks_Tensorboard.ipynb**: Using callbacks and TensorBoard for model monitoring

#### B. Tuning
- **Keras_Tuner.ipynb**: Hyperparameter optimization using Keras Tuner

#### C. Data Augmentation
This section explores various data augmentation techniques across different data modalities:

- **Image_Augmentation/**
  - TF_Image_Augmentation.ipynb: Standard TensorFlow image augmentation techniques
  - KerasCV_Augmentation.ipynb: Advanced image augmentation using KerasCV
  - FastAI_Augmentation.ipynb: Image augmentation using FastAI library
- **Text_Augmentation.ipynb**: NLP augmentation techniques using nlpaug
- **Time_Series_Augmentation.ipynb**: Augmentation methods for time series data
- **Tabular_Data_Augmentation.ipynb**: Techniques for tabular data augmentation
- **Speech_Augmentation.ipynb**: Audio data augmentation methods
- **Document_Image_Augmentation.ipynb**: Specialized techniques for document images

### Part 2: Advanced Keras Components

**Advanced_Keras_Constructs.ipynb** demonstrates numerous advanced Keras features:

1. Custom learning rate scheduler (OneCycleScheduler)
2. Custom dropout implementation
3. Custom normalization (MaxNormDense)
4. TensorBoard integration for visualization
5. Custom loss function (Huber Loss)
6. Custom activation functions, initializers, regularizers, and constraints
7. Custom metrics implementation
8. Custom layers development
9. Custom model architecture (Residual Networks)
10. Custom optimizer implementation
11. Custom training loop

## Video Demonstrations

A detailed walkthrough video explaining each notebook is available [here](#) (link to be updated).

## Requirements

The notebooks require the following libraries:
- TensorFlow 2.x
- Keras
- KerasCV
- FastAI
- nlpaug
- AugLy
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## Usage

1. Clone this repository
2. Install the required dependencies
3. Open the notebooks in Google Colab or Jupyter
4. Execute the cells in sequence

## References

- [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://github.com/ageron/handson-ml3)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Data Augmentation Review](https://github.com/AgaMiko/data-augmentation-review)
- [AugLy Library](https://github.com/facebookresearch/AugLy)
- [FastAI Library](https://github.com/fastai/fastbook)
