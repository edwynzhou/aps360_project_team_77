# Chest X-Ray Imaging Neural Network | APS360 Project

### Project
Human health relies on timely and efficient medical diagnosis, particularly for conditions that require rapid intervention. Pneumonia, a severe respiratory infection that targets the lungs, remains a major health concern, particularly in pediatric and immunocompromised patients. Early detection pneumonia is crucial for effective treatment, yet traditional diagnostic methods- such as expert radiologists analyzing patient chest x-rays- can be time-consuming and subject to interobserver variability, where different healthcare providers interpret the same chest x-ray differently. With the growing availability of medical imaging data, machine learning techniques such as deep learning offer a promising solution to automating key aspects of the diagnostic process.

Our project explores the application of deep learning techniques to classify chest x-ray images into normal and pneumonia-positive patients. The goal of this model is not to replace expert radiologists in pneumonic diagnosing, but rather, to provide a preliminary screening tool that flags potentially concerning cases, allowing for faster prioritization and review by medical professionals. 

To accomplish this, we leveraged the efficiency of a Convolutional Neural Network (CNN) to extract spatial and geometric features from the chest x-ray images. Passing these processed features to a fully connected Artificial Neural Network (ANN) will transform it into a linearly separable problem, facilitating classification. The use of neural networks in this way offers a scalable and efficient approach to medical imaging analysis, without overlooking the indispensable role of human expertise in final diagnosis decisions. 

### Data Processing
To process the dataset, we used torchvision.transforms to create a pipeline that resizes, crops, and normalizes all the images in the dataset. In doing so, we cleaned the data to be the same dimensions (224x224x3), and cropped the images to increase the robustness of the CNN. The following is the data transformation process:
The images were resized to 256 pixels to reduce computational cost as original images are mostly at least 1000 pixels in length and width
Images were center cropped to increase robustness of the CNN while also reducing image size to 224 by 224 to decrease computational cost
Images were then normalized by their mean and standard deviation 
The dataset was also unbalanced, which initially resulted in low validation error despite high training error, and eventually reached 0 error in both training and validation in less epochs than anticipated as a result. The classes were balanced from 1583 normal and 4273 pneumonia samples to 1440 in each class by intentionally deleting files until there were each class. In the future, we plan to perform random deletion to avoid biases in removing data.

<img width="430" alt="Screenshot 2025-05-15 at 11 33 07 AM" src="https://github.com/user-attachments/assets/ae3e2092-79c8-413f-9089-5f1588f7118a" />

### Baseline SVM Model

To establish a baseline for comparison, we will implement a Support Vector Machine (SVM) with a linear kernel to classify our data into two labels: normal or pneumonia-positive. The SVM will identify the optimal hyperplane that maximizes the margin between the closest data points of each class, effectively separating them. By fitting the data to this model, the SVM will learn a decision boundary based on the discovered hyperplane, providing a foundational benchmark for evaluating our approach.

SVM Data Processing
Since SVM’s require two-dimensional input and our processed data was originally three-dimensional, the data was reshaped accordingly. To address this, all of the images were converted to uniform size and then transformed into NumPy arrays. Data was flattened to a two-dimensional array to make it compatible with the SVM model. Additionally, the same 80/10/10 training split was maintained as the CNN models, ensuring consistency across model  hi grace, are you done with your section? evaluations. 

Training the SVM
To train the baseline model, the Scikit-Learn’s Support Vector Classifier (SVC) was used with a linear kernel to distinguish between normal and pneumonia-positive cases. The dataset was split into training and test subsets using the 80/10/10 distribution as previously justified. After training the model on the preprocessed dataset, the model performance was evaluated by comparing the predicted labels with the ground truth labels using the Scikit-Learn’s metrics library. It was found that the model achieved a training accuracy of 93.33%, with an error of 6.67%. Figure 5 shows the confusion matrix that highlights the model’s performance, detailing the number of correctly and incorrectly classified samples across both classes.

<img width="474" alt="Screenshot 2025-05-15 at 11 35 49 AM" src="https://github.com/user-attachments/assets/38453f30-fa0c-43d1-8c72-8ba3207e6990" />

Visualizing the Decision Boundary
To visualize the decision boundary and the classification results of our SVM model, we applied Principle Component Analysis (PCA) to reduce the dataset’s dimensionality and project it onto a two-dimensional plane. Since PCA preserves linear trends in the data, it provides a useful approximation of how the SVM’s hyperplane separates the boundary of the two classes. Figure 6 below illustrates the decision boundary, along with the support vectors that influence it. Due to computational constraints, the grid size was reduced, meaning some data points are not fully represented in the visualization. Despite this limitation, the PCA projection offers valuable insight into how the model was able to differentiate between the two classes.

<img width="631" alt="Screenshot 2025-05-15 at 11 36 34 AM" src="https://github.com/user-attachments/assets/4a7861ce-3bb8-4197-99e3-694a11e17f2d" />

### Primary Model
<img width="641" alt="Screenshot 2025-05-15 at 11 38 47 AM" src="https://github.com/user-attachments/assets/9ed0cd48-cb4f-4f5f-9983-0186d5b5cd9e" />




### Challenges

The team faced significant challenges related to data processing. Beyond issues outlined in section 3, our dataset was over 3000 images, making it very computationally expensive to fine-tune the model’s hyperparameters through the iterative process outlined in section 5. Hence, we decided to reduce the training, validation and testing datasets for this phase, allowing us to broadly explore hyperparameter configurations while maintaining sufficient data to show the project’s progress.



## Data
- Datasets retrieved from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
- new test data retrieved from https://www.kaggle.com/datasets/esfiam/balanced-chest-x-ray-dataset
### Available at:
- https://drive.google.com/drive/folders/1KJX4PYt3Eix23UrOvm7snrxcdzb6dTx9?usp=sharing
- https://drive.google.com/drive/folders/1UoGvSv09gopZh4eAw9zgo0sekzxOVqjj?usp=sharing
