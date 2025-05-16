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

#### SVM Data Processing
Since SVM’s require two-dimensional input and our processed data was originally three-dimensional, the data was reshaped accordingly. To address this, all of the images were converted to uniform size and then transformed into NumPy arrays. Data was flattened to a two-dimensional array to make it compatible with the SVM model. Additionally, the same 80/10/10 training split was maintained as the CNN models, ensuring consistency across model  hi grace, are you done with your section? evaluations. 

#### Training the SVM
To train the baseline model, the Scikit-Learn’s Support Vector Classifier (SVC) was used with a linear kernel to distinguish between normal and pneumonia-positive cases. The dataset was split into training and test subsets using the 80/10/10 distribution as previously justified. After training the model on the preprocessed dataset, the model performance was evaluated by comparing the predicted labels with the ground truth labels using the Scikit-Learn’s metrics library. It was found that the model achieved a training accuracy of 93.33%, with an error of 6.67%. Figure 5 shows the confusion matrix that highlights the model’s performance, detailing the number of correctly and incorrectly classified samples across both classes.

<img width="474" alt="Screenshot 2025-05-15 at 11 35 49 AM" src="https://github.com/user-attachments/assets/38453f30-fa0c-43d1-8c72-8ba3207e6990" />

#### Visualizing the Decision Boundary
To visualize the decision boundary and the classification results of our SVM model, we applied Principle Component Analysis (PCA) to reduce the dataset’s dimensionality and project it onto a two-dimensional plane. Since PCA preserves linear trends in the data, it provides a useful approximation of how the SVM’s hyperplane separates the boundary of the two classes. Figure 6 below illustrates the decision boundary, along with the support vectors that influence it. Due to computational constraints, the grid size was reduced, meaning some data points are not fully represented in the visualization. Despite this limitation, the PCA projection offers valuable insight into how the model was able to differentiate between the two classes.

<img width="631" alt="Screenshot 2025-05-15 at 11 36 34 AM" src="https://github.com/user-attachments/assets/4a7861ce-3bb8-4197-99e3-694a11e17f2d" />

### Primary Model
The team designed a model that integrates a CNN and ANN in order to classify input images into normal or pneumonia-positive. 

<img width="641" alt="Screenshot 2025-05-15 at 11 38 47 AM" src="https://github.com/user-attachments/assets/9ed0cd48-cb4f-4f5f-9983-0186d5b5cd9e" />


The model accepts a 3D tensor with dimensions 3 × 224 × 244 (channels × height × width) and applies 2D-convolutional filters (kernels) to extract meaningful features interpretable by the model. Following each convolution, max pooling kernel consolidates the features extracted and passes the output to another convolutional layer. This process is repeated multiple times to progressively refine the features and emphasize the most relevant information. At this point, the features extracted by the CNN are sufficiently relevant to make the classification problem linearly separable. Thus, the features are flattened to one dimension and fed to a fully connected network (ANN) to classify the image. 

To optimize the model’s parameters (weights and biases) during training, a Cross Entropy Loss function is used and a Steepest Gradient Descent function with momentum set to 0.9 to ensure convergence while avoiding getting stuck on local minima, enhancing learning efficiency.

#### Hyperparameter tuning
To optimize model performance, the team iteratively tested various hyperparameters, including the number of convolutional and pooling layers, batch size, learning rate and number of fully connected layers. Three different configurations were evaluated. After each iteration, model performance was recorded and hyper parameters were fine-tuned to improve outcomes for the next iteration. The results of this process may be found in the table below.

<img width="633" alt="Screenshot 2025-05-15 at 11 40 36 AM" src="https://github.com/user-attachments/assets/8995b697-4608-4d16-af5d-399618eb8bcd" />

#### Best Primary Model
After this exploration and based on the smallest training and validation error, the team picked the following model as the best primary model.

* Three convolutional layers: For small classification problems, deep networks might overcomplicate and overfit the dataset, while shallow networks might underfit complex data. Three convolutional layers strike a balance between feature extraction capability and computational cost, while simultaneously avoiding the network from being too deep and risking overfitting or Vanishing/Exploding gradients. * First layer applies 16 filters, second applies 32, third applies 56.
  * Feature space is reduced from 224 →110 → 53 → 24. 
  * Kernel size 5×5: High-dimensional kernels are computationally expensive and might miss important features due to their high span over the input data. Low-dimensional kernels may not be receptive enough to important features, hence failing to capture relevant information. For medical imaging, 5×5 kernels ensure that features like lung-opacities that extend over several pixels are detected.
* Max-pooling 2×2 with Stride 2: Big pooling kernels Max pooling is used to retain the highest activations and thus the most relevant data of the extracted feature. Big pooling kernels may ignore subtleties in the feature space that may be relevant for the classification task later on. Hence, a 2×2 pooling kernel with stride 2 efficiently reduces the feature space by approximately half while decreasing computational cost, memory usage, and increasing focus on dominant patterns. 
* Batch size of 54: Small batch sizes lead to gradient noise, while large batches though smoother come with a high memory cost. A batch size of 54 offers a balance between gradient stability and memory usage, allowing for reasonably fast optimization updates without excessive noise.
* Learning rate 0.015: Small learning rates decrease learning speed and may lead to getting stuck on local minima, while big learning rates increase learning speed but at the cost of unstable convergence and possibly overfitting the dataset. This learning rate, though higher than conventional models, is a perfect fit for our choice of optimization function (SGD with momentum), as it ensures convergence without being too big to overshoot and go past a local minima.
* Two fully-connected layers (ANN): Though shallow, a two-layer ANN with high width (1st layer 512 neurons, 2nd layer 2 neurons) proved to be enough to solve the problem, meaning that the CNN extracted enough features with sufficient detail for the problem to be linearly separable and easily solved using only 2 layers of connected neurons. 


### Model Performance

#### Verifying learning ability

The model was trained on a sample of 20 images over 30 epochs (Batch size =12, learning rate = 0.01) to test overfitting. As seen in the figure below, the model reached 0% training error and loss, meaning that memorization of the small training dataset was achieved. Additionally, we can note a 50% validation error and a spike in validation loss at the end of Epoch 30, both very high and showcasing poor performance, indicative of overfitting.

<img width="654" alt="Screenshot 2025-05-16 at 6 04 45 PM" src="https://github.com/user-attachments/assets/b76143cb-7893-466a-abca-9e629dc3b8b0" />

#### Model performance on test data

The model chosen and described in 5.3 was trained and validated using the 80/10/10 split described in section 3. The image below shows the results of the training.

<img width="665" alt="Screenshot 2025-05-16 at 6 07 24 PM" src="https://github.com/user-attachments/assets/7b3d207c-c77e-431e-a30a-8ca75ef319d2" />


As we can see, spikes in validation error and loss may be indicative of overshooting, product of a big learning rate or batch size. However, after epoch 20, the model reaches convergence stability. Similarly, there are no spikes after epoch 20 and no big difference between validation and training error, which suggests there was no overfitting.

<img width="418" alt="Screenshot 2025-05-16 at 6 07 55 PM" src="https://github.com/user-attachments/assets/73d63c9c-cc9f-4d0b-bd2c-aff75c844d12" />


After testing the model against the test dataset, Type I and Type II errors arise as seen in the confusion matrix. Type I errors (False Positive: 21.43% rate), are most likely caused by the model identifying a pattern in the “Normal” X-ray image that it deemed important in the Pneumonia case, possibly linked to image quality or patients having conditions that mimic Pneumonia on X-rays. Type II errors (False Negatives: 50% rate) may be caused by the model not picking up on subtle features showcased on Pneumonia-positive X-rays. 


### Challenges

The team faced significant challenges related to data processing. Beyond issues outlined in section 3, our dataset was over 3000 images, making it very computationally expensive to fine-tune the model’s hyperparameters through the iterative process outlined in section 5. Hence, we decided to reduce the training, validation and testing datasets for this phase, allowing us to broadly explore hyperparameter configurations while maintaining sufficient data to show the project’s progress.



## Data
- Datasets retrieved from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
- new test data retrieved from https://www.kaggle.com/datasets/esfiam/balanced-chest-x-ray-dataset
### Available at:
- https://drive.google.com/drive/folders/1KJX4PYt3Eix23UrOvm7snrxcdzb6dTx9?usp=sharing
- https://drive.google.com/drive/folders/1UoGvSv09gopZh4eAw9zgo0sekzxOVqjj?usp=sharing
