# Retinal Disease Classification using Swin Transformer

## Overview
This project implements a deep learning pipeline to classify retinal diseases using the Swin Transformer model. The dataset consists of Optical Coherence Tomography (OCT) images, which are preprocessed and used to train a robust image classification model.

## Features
- Dataset preprocessing with augmentation
- Dataset statistics computation (mean, standard deviation)
- Class balancing through undersampling
- Implementation of Swin Transformer for classification
- GPU acceleration support (if available)
- Model evaluation using accuracy, precision, and confusion matrix
- Model checkpointing and saving

## Dataset
The dataset is structured as follows:
```
OCT/
├── train/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
```

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Training the Model
To train the Swin Transformer model, run the following command:
```bash
python train.py
```

## Evaluating the Model
To evaluate the model on the test dataset, use:
```bash
python evaluate.py
```

## Model Saving
The trained model is saved as `swin_transformer_retinal_disease.pkl` and can be loaded for inference.

## Results
After training, the model's performance is evaluated using accuracy, precision, and a confusion matrix.

## License
This project is open-source and available under the MIT License.





For Dataset visti: https://data.mendeley.com/datasets/rscbjbr9sj/3
