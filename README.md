# Chest X-Ray Imaging Neural Network | APS360 Project

Human health relies on timely and efficient medical diagnosis, particularly for conditions that require rapid intervention. Pneumonia, a severe respiratory infection that targets the lungs, remains a major health concern, particularly in pediatric and immunocompromised patients. Early detection pneumonia is crucial for effective treatment, yet traditional diagnostic methods- such as expert radiologists analyzing patient chest x-rays- can be time-consuming and subject to interobserver variability, where different healthcare providers interpret the same chest x-ray differently. With the growing availability of medical imaging data, machine learning techniques such as deep learning offer a promising solution to automating key aspects of the diagnostic process.

Our project explores the application of deep learning techniques to classify chest x-ray images into normal and pneumonia patients. The goal of this model is not to replace expert radiologists in pneumonic diagnosing, but rather, to provide a preliminary screening tool that flags potentially concerning cases, allowing for faster prioritization and review by medical professionals. 

To accomplish this, we leveraged the efficiency of a Convolutional Neural Network (CNN) to extract spatial and geometric features from the chest x-ray images. Passing these processed features to a fully connected Artificial Neural Network (ANN) will transform it into a linearly separable problem, facilitating classification. The use of neural networks in this way offers a scalable and efficient approach to medical imaging analysis, without overlooking the indispensable role of human expertise in final diagnosis decisions. 



## Data
- Datasets retrieved from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download
- new test data retrieved from https://www.kaggle.com/datasets/esfiam/balanced-chest-x-ray-dataset
### Available at:
- https://drive.google.com/drive/folders/1KJX4PYt3Eix23UrOvm7snrxcdzb6dTx9?usp=sharing
- https://drive.google.com/drive/folders/1UoGvSv09gopZh4eAw9zgo0sekzxOVqjj?usp=sharing
