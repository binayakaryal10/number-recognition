# 🔢 Real-Time Handwritten Digit Recognition (CNN + Camera & Streamlit Web App)

A complete end-to-end Deep Learning system that recognizes handwritten numbers using a Convolutional Neural Network (CNN) trained on the MNIST dataset (**99.09% test accuracy**).

This project supports **two deployment environments**:
1. **🌐 Web Application (Streamlit)**: Deployable online so anyone with a smartphone or browser can snap a photo, upload an image, or draw digits on an interactive canvas.
2. **💻 Real-Time Desktop Camera Feed (OpenCV)**: High-speed live video stream with HUD, crosshair targeting ROI, and live probability distribution.

---

## 🚀 1. Run the Web Application (Streamlit)

### Run Locally (1-Click)
Double-click:
```
run_streamlit.bat
```
Or run from terminal:
```powershell
C:\Users\LEGION\anaconda3\envs\tf_env\Scripts\streamlit.exe run app.py
```
Then open **`http://localhost:8501`** in your web browser!

### Web App Features:
- 📸 **Camera Snap**: Use your laptop webcam or smartphone camera to capture a photo of a digit on paper.
- 🖌️ **Interactive Canvas**: Draw digits directly on screen using your mouse, trackpad, or finger.
- 📁 **Image Upload**: Upload any PNG/JPG image containing handwritten numbers.
- 🧠 **Explainable Visuals**: View a side-by-side comparison of your raw photo vs. the normalized $28 \times 28$ image that the CNN sees.
- 📊 **Probability Bar Chart**: Live softmax probabilities across all 10 digits (0 to 9).

---

## 🌍 2. Deploy to the Public Web (Free for Anyone to Use)

You can share your project with friends, recruiters, or users worldwide using **Streamlit Community Cloud** (100% free hosting):

### Step-by-Step Deployment:
1. **Push your code to GitHub**:
   - Initialize git in this directory (exclude large CSVs like `mnist_train.csv` via `.gitignore`):
     ```bash
     git init
     git add app.py preprocess_utils.py mnist_cnn_model.keras requirements.txt README.md
     git commit -m "Deploy MNIST CNN Streamlit App"
     git branch -M main
     git remote add origin https://github.com/YOUR_USERNAME/digit-recognizer.git
     git push -u origin main
     ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)**:
   - Sign in with your GitHub account.
3. **Click "New App"**:
   - Select your repository: `digit-recognizer`
   - Select Branch: `main`
   - Main file path: `app.py`
4. **Click "Deploy!"**:
   - Streamlit Cloud will read `requirements.txt`, install dependencies, load `mnist_cnn_model.keras`, and give you a public URL (e.g. `https://yourname-digit-recognizer.streamlit.app`)!
   - Now anyone on any phone or laptop can visit the link, enable their camera, and recognize digits!

---

## 💻 3. Run the Desktop Real-Time Camera Feed (OpenCV)

### Run Locally (1-Click)
Double-click:
```
run_camera.bat
```
Or run from terminal:
```powershell
C:\Users\LEGION\anaconda3\envs\tf_env\python.exe realtime_detector.py
```

### Desktop Camera Controls:
| Key | Action |
| :--- | :--- |
| **`q`** or **`ESC`** | Quit application |
| **`m`** | Toggle Mode: Center Target ROI Box vs. Multi-Digit Auto-Contour Detection |
| **`t`** | Toggle binary threshold mask view |
| **`p`** | Pause / Freeze live frame |
| **`s`** | Save a snapshot to disk (`digit_snapshot_X.png`) |

---

## 🧠 Preprocessing Pipeline (Why it Works on Real-World Paper)

Unlike raw camera crops which fail on MNIST models, this project implements Yann LeCun's exact centering standard:
1. **Adaptive Inverted Thresholding**: Converts dark ink on white paper into bright strokes on a dark background.
2. **Noise Filtering & Dilation**: Cleans speckles and thickens fine ballpoint pen lines to match MNIST stroke weight.
3. **Aspect-Ratio Preserving Resize**: Resizes the bounding box to fit inside $20 \times 20$ pixels without stretching.
4. **Center-of-Mass Centering**: Computes spatial moments ($M_{10}/M_{00}$ and $M_{01}/M_{00}$) and shifts the digit directly to the center of a $28 \times 28$ canvas.
5. **Normalization**: Scales pixels to $[0.0, 1.0]$ for the CNN input tensor.

---

## 📂 Project Structure

```
Number recognization/
├── app.py                  # Streamlit Web Application (Camera + Canvas + Upload)
├── realtime_detector.py    # OpenCV Desktop live camera application
├── preprocess_utils.py     # Adaptive thresholding & center-of-mass centering
├── train_model.py          # Script to train CNN and export .keras model
├── mnist_cnn_model.keras   # Saved trained CNN model (~1.4 MB, 99.09% accuracy)
├── requirements.txt        # Cloud deployment dependencies (Streamlit Cloud, etc.)
├── run_streamlit.bat       # 1-Click launcher for Streamlit Web App
├── run_camera.bat          # 1-Click launcher for OpenCV Camera App
├── README.md               # Documentation & deployment guide
├── untitled.ipynb          # Original exploratory notebook
├── mnist_train.csv         # MNIST training dataset
└── mnist_test.csv          # MNIST test dataset
```
