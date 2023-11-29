# Dog breed classification

## Introduction
This project is a Python-based image classification application that identifies and then classifies a dog breed in an image. It uses a YOLOv5 model to detect the dog in the image and then a Vision Transformer (ViT) to classify the dog breed accurately.

## Features
- **Object Detection**: Utilizes YOLOv5 for detecting dogs in images.
- **Dog Breed Classification**: Employs a Vision Transformer (ViT) for accurate classification of dog breed.
- **Efficient Handling**: Supports image inputs from both local paths and URLs.

## Prerequisites
- Python 3.x
- Libraries: `transformers`, `torchvision`, `datasets`, `Pillow`, `PyTorch`, `NumPy`,`torchmetrics`,`NumPy`

## Installation
1. Clone the repository: `git clone [repository-link]`.
2. Install required Python packages: `pip install -r requirements.txt`.

## Usage
To use this dog breed detection system:
1. Train the ViT on the dataset of your choice. I used [this dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). I also [created a google collab notebook for easy training.](https://colab.research.google.com/drive/1vg3G4hzn8C-l3JlMXBzPLo1V7WbX0WU3?usp=sharing)
2. Place the newly created PetModel folder in the same directory as `predictions.py`
3. Run `python predictions.py image` where "image" is a URL or path to a JPG file on your computer.
## Authors and Acknowledgment
- Noah Mitchem - Code implementation
## 
![](http://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNmVhYXhhbnNqOGRlNnBvcmo1encyNmJjeHJ4NG54bDQ4eTRsczlvaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/BX3dvmYAGWNApdCSug/giphy.gif)
