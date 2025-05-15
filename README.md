# Chest X-Ray Imaging Neural Network | APS360 Project

### Project
Human health relies on timely and efficient medical diagnosis, particularly for conditions that require rapid intervention. Pneumonia, a severe respiratory infection that targets the lungs, remains a major health concern, particularly in pediatric and immunocompromised patients. Early detection pneumonia is crucial for effective treatment, yet traditional diagnostic methods- such as expert radiologists analyzing patient chest x-rays- can be time-consuming and subject to interobserver variability, where different healthcare providers interpret the same chest x-ray differently. With the growing availability of medical imaging data, machine learning techniques such as deep learning offer a promising solution to automating key aspects of the diagnostic process.

Our project explores the application of deep learning techniques to classify chest x-ray images into normal and pneumonia patients. The goal of this model is not to replace expert radiologists in pneumonic diagnosing, but rather, to provide a preliminary screening tool that flags potentially concerning cases, allowing for faster prioritization and review by medical professionals. 

To accomplish this, we leveraged the efficiency of a Convolutional Neural Network (CNN) to extract spatial and geometric features from the chest x-ray images. Passing these processed features to a fully connected Artificial Neural Network (ANN) will transform it into a linearly separable problem, facilitating classification. The use of neural networks in this way offers a scalable and efficient approach to medical imaging analysis, without overlooking the indispensable role of human expertise in final diagnosis decisions. 

### Data Processing
To process the dataset, we used torchvision.transforms to create a pipeline that resizes, crops, and normalizes all the images in the dataset. In doing so, we cleaned the data to be the same dimensions (224x224x3), and cropped the images to increase the robustness of the CNN. The following is the data transformation process:
The images were resized to 256 pixels to reduce computational cost as original images are mostly at least 1000 pixels in length and width
Images were center cropped to increase robustness of the CNN while also reducing image size to 224 by 224 to decrease computational cost
Images were then normalized by their mean and standard deviation 
The dataset was also unbalanced, which initially resulted in low validation error despite high training error, and eventually reached 0 error in both training and validation in less epochs than anticipated as a result. The classes were balanced from 1583 normal and 4273 pneumonia samples to 1440 in each class by intentionally deleting files until there were each class. In the future, we plan to perform random deletion to avoid biases in removing data.
<img width="430" alt="Screenshot 2025-05-15 at 11 33 07 AM" src="https://github.com/user-attachments/assets/ae3e2092-79c8-413f-9089-5f1588f7118a" />



### Baseline SVM Model

## Data
- Datasets retrieved from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
- new test data retrieved from https://www.kaggle.com/datasets/esfiam/balanced-chest-x-ray-dataset
### Available at:
- https://drive.google.com/drive/folders/1KJX4PYt3Eix23UrOvm7snrxcdzb6dTx9?usp=sharing
- https://drive.google.com/drive/folders/1UoGvSv09gopZh4eAw9zgo0sekzxOVqjj?usp=sharing
