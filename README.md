# Arabic Text Classification using MNIST Dataset

## Overview
This project focuses on the classification of Arabic text using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The dataset used for training and testing the model is an Arabic text dataset in the MNIST format.

## Project Structure
The project is organized as follows:

- **Data Preparation**: Extracting the dataset from a RAR file, splitting it into training and validation sets, and organizing the data into respective folders.
- **Data Augmentation**: Using the `ImageDataGenerator` to rescale the images and generate batches of tensor image data for training and validation.
- **Model Architecture**: Implementing CNN models with varying layers and configurations to evaluate performance.
- **Training**: Compiling and training the models with training and validation data.
- **Evaluation**: Plotting the accuracy of the models to compare performance across different architectures.

## Data Preparation
1. **Extracting Dataset**: The dataset is extracted from a RAR file and organized into a structured format with separate folders for each class.
2. **Splitting Data**: The dataset is split into training and validation sets with an 80-20 split ratio.
3. **Organizing Data**: The images are moved to respective training and validation folders based on their class.

## Data Augmentation
Data augmentation is performed using the `ImageDataGenerator` class from Keras. The images are rescaled and batches of tensor image data are generated for training and validation.

## Model Architecture
Three different CNN models with varying layers and configurations are implemented to classify the Arabic text:

1. **Model with 1 Convolutional Layer**
2. **Model with 2 Convolutional Layers**
3. **Model with 3 Convolutional Layers**

Additionally, a model with 3 convolutional layers and dropout is also implemented to evaluate the impact of dropout on model performance.

## Training
The models are compiled using the RMSprop optimizer and sparse categorical crossentropy loss function. Each model is trained on the training data and validated using the validation data for a fixed number of epochs.

## Evaluation
The performance of the models is evaluated by plotting the training and validation accuracy across epochs. This helps in comparing the models and selecting the best-performing one.

## Results
The accuracy plots for training and validation are generated to visualize the performance of the models. The models are saved for future use and comparison.

## Conclusion
This project demonstrates the process of building, training, and evaluating CNN models for Arabic text classification using the MNIST dataset format. The results provide insights into the effectiveness of different model architectures and the impact of dropout on model performance.

## Requirements
- TensorFlow
- Keras
- Matplotlib
- Numpy
- OS
- Rarfile
- Shutil

## How to Run
1. **Extract Dataset**: Ensure the dataset is in a RAR file and extract it using the provided function.
2. **Prepare Data**: Split the dataset into training and validation sets and organize them into respective folders.
3. **Train Models**: Train the CNN models using the training data and validate using the validation data.
4. **Evaluate Models**: Plot the training and validation accuracy to compare the models.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:

- Fork the repository
- Create a new branch (git checkout -b feature-branch)
- Make your changes
- Commit your changes (git commit -m 'Add new feature')
- Push to the branch (git push origin feature-branch)
- Open a pull request
## Issues
If you encounter any issues or have any questions about the project, please feel free to open an issue in the GitHub repository.

