# 🔢 Handwritten Digit Recognition

A handwritten digit recognition project built using **Python, TensorFlow/Keras, CNN, OpenCV, and Streamlit**.

The model is trained on the **MNIST dataset** and achieved **99.09% accuracy on the MNIST test set**.

The project includes a Streamlit web app where you can draw a digit, upload an image, or use a camera to make a prediction.

> **Note:** The model performs best when the digit is clear, centered, and similar to the images in the MNIST dataset. Real-world photos can be more difficult because of differences in handwriting, lighting, background, and image quality.

## 🚀 Features

* ✍️ Draw a digit directly on the web app
* 📷 Capture a digit using a camera
* 📁 Upload an image of a handwritten digit
* 🔢 Predict digits from **0 to 9**
* 📊 View the prediction probabilities
* 🖼️ See how the uploaded image is processed before being given to the model
* 💻 Run a real-time camera version locally using OpenCV

## 🧠 Model

The project uses a **Convolutional Neural Network (CNN)** trained on the **MNIST handwritten digit dataset**.

**Test Accuracy:** 99.09%

The model expects an image similar to the MNIST format:

* Image size: `28 × 28`
* Grayscale image
* Pixel values normalized between `0` and `1`

## 🖼️ Image Preprocessing

Images from a camera or uploaded by the user are not always in the same format as MNIST images. To make them easier for the model to recognize, the project applies several preprocessing steps:

1. Convert the image to grayscale
2. Apply thresholding to separate the digit from the background
3. Remove small amounts of noise
4. Find the digit in the image
5. Resize it while keeping its proportions
6. Place the digit near the center of a `28 × 28` image
7. Normalize the pixel values

This preprocessing makes real-world handwritten digits more similar to the images used during training.

## 🌐 Streamlit Web App

The project is deployed using **Streamlit**, so it can be accessed from a browser without running the code locally.

The web app supports:

* **Drawing:** Draw a digit using your mouse, trackpad, or finger.
* **Camera:** Capture a handwritten digit using a camera.
* **Image Upload:** Upload a PNG or JPG image.
* **Prediction:** Get the predicted digit along with probabilities for all 10 digits.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/binayakaryal10/number-recognition.git
cd number-recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

## 📷 Run the Desktop Camera Version

You can also run the OpenCV camera application:

```bash
python realtime_detector.py
```

Or, on Windows, you can use:

```text
run_camera.bat
```

### Camera Controls

| Key         | Action                |
| ----------- | --------------------- |
| `Q` / `ESC` | Quit                  |
| `M`         | Change detection mode |
| `T`         | Toggle threshold view |
| `P`         | Pause the camera      |
| `S`         | Save a snapshot       |

## 📁 Project Structure

```text
number-recognition/
│
├── app.py                  # Streamlit web application
├── realtime_detector.py    # OpenCV real-time camera application
├── preprocess_utils.py     # Image preprocessing functions
├── train_model.py          # CNN training script
├── mnist_cnn_model.keras   # Trained CNN model
├── requirements.txt        # Python dependencies
├── packages.txt            # Additional deployment packages
├── runtime.txt             # Python runtime version
├── run_streamlit.bat       # Windows Streamlit launcher
├── run_camera.bat          # Windows camera launcher
├── untitled.ipynb          # Original notebook
└── README.md               # Project documentation
```

## 📚 What I Learned

Working on this project helped me understand:

* How CNNs are used for image classification
* How to train and save a TensorFlow/Keras model
* Image preprocessing with OpenCV
* Working with the MNIST dataset
* Connecting a trained model to a web application
* Deploying a machine learning project using Streamlit
* The difference between performance on a standard dataset and real-world images

## 🔗 Project

**GitHub:**
https://github.com/binayakaryal10/number-recognition

**Live Demo:**
https://binayak-numberrecognition.streamlit.app/

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* MNIST
