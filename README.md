# Facial Emotion Recognition using CNN

This project focuses on recognizing human emotions from facial expressions using a Convolutional Neural Network (CNN). Facial Emotion Recognition (FER) is a key application in artificial intelligence and computer vision, with use cases in human-computer interaction, mental health monitoring, and customer feedback analysis.

## Project Notebook
The full implementation can be accessed on Google Colab:  
[Click to View Notebook](https://colab.research.google.com/drive/1fzZC3MKsszhtbEnIIpYVu2XL4Yo4D7n4?usp=sharing)

## Dataset
The project uses the FER-2013 dataset consisting of:
- 48x48 grayscale facial images
- 7 emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### Data Files
- `train.csv`: Contains pixel values and emotion labels (0–6)
- `test.csv`: Contains only pixel values for prediction

## Technologies Used
- TensorFlow / Keras – Deep learning framework
- OpenCV – Image processing
- NumPy, Pandas – Data handling
- Matplotlib, Seaborn – Visualization
- Scikit-learn – Evaluation metrics

## Model Architecture
- Convolutional Layers for spatial feature extraction
- Pooling Layers to reduce dimensionality
- Fully Connected Layers for classification
- Activation Functions: ReLU (hidden layers), Softmax (output layer)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

## Workflow
1. Load and preprocess the data
2. Build the CNN model
3. Train the model using training data
4. Evaluate model using accuracy, precision, recall, and F1-score
5. Test on new images or video for real-time prediction

## Output Analysis
- Training/Validation accuracy and loss curves
- Confusion matrix for emotion classification
- Real-time predictions on sample inputs

## Possible Improvements
- Incorporate transfer learning using pretrained models
- Expand training with larger datasets
- Deploy as a web/mobile application
- Tune hyperparameters for improved accuracy

## References
1. Mehendale, N. (2020). Facial Emotion Recognition using CNN (FERC)
2. Agung et al. (2024). Image-based FER using Emognition dataset
3. Procedia Computer Science (2024). Real-time FER using CNN

## License
This project is developed for academic purposes under the Deep Learning course at Gitam University.
