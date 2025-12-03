# CNN_CIFAR-10-Image-classifier_GPU
A Convolution neural network on CIFAR-10 data Image classifier using a GPU. Built and trained AI model with unseen data for scientific computation using real world image. With real time loss and accuracy updates along with measuring quantitative metrics such as confusion matrix.
- This project demonstrates how to build, train, and benchmark a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.
- It highlights key concepts in scientific computing, including GPU acceleration, device management, and performance measurement.

## Project Overview
This project covers:
- Building a CNN model from scratch using PyTorch
- Training the model on the CIFAR-10 dataset
- Running inference on both CPU and GPU
- Benchmarking and comparing CPU vs GPU computation time
- Testing the trained model on real-world images
- Handling device compatibility (cpu vs cuda) to avoid runtime errors
- This project is implemented in Google Colab, making it runnable even for users without local GPU support.

## Technologies Used
- Python
- PyTorch
- Google Colab
- CUDA / GPU acceleration
- Torchvision (datasets + transforms)
- Numpy, PIL, Matplotlib

## Key Features
- 1. Model Development
   - Implemented a custom CNN for CIFAR-10
   - Used convolution, ReLU, pooling, and fully connected layers
   - Applied CrossEntropyLoss and Adam optimizer

- 2. GPU-Accelerated Scientific Computing
   - Leveraged CUDA for high-performance training and inference
   - Compared timings between CPU and GPU using: torch.cuda.synchronize()

3. Device Management
   Prevented common errors like: RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
   by using:  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X = X.to(device)

4. Real-World Image Testing
- Loaded external images with PIL
- Applied CIFAR-10 preprocessing transforms
- Ran inference to validate real-world performance

## Results
- Training on GPU was significantly faster than CPU
- The trained model achieved competitive accuracy on CIFAR-10
