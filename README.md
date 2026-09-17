Intrusion Detection Using Wireless Network and Machine Learning

A real-time **Intrusion Detection System (IDS)** designed for wireless IoT networks using Machine Learning and Deep Learning techniques.

The system uses a **Hybrid Autoencoder–Attention LSTM model** to extract important network features, capture sequential patterns, and identify abnormal or malicious network traffic. The application is deployed using the **Flask web framework**. The project uses the **IoT-23 dataset** for training and evaluation.

📌 Project Overview

The system performs the following major tasks:

* Collects IoT network traffic data
* Performs data preprocessing and cleaning
* Applies feature scaling and feature selection
* Uses SMOTE to handle imbalanced data
* Uses an Autoencoder for feature extraction and dimensionality reduction
* Uses Attention-based LSTM for sequential pattern analysis
* Detects suspicious network activity
* Provides a Flask-based web interface for prediction
* Displays the intrusion detection result

The project architecture and modules are based on data collection, preprocessing, feature selection, model training, and intrusion detection.

🛠️ Technologies Used

Backend

* Python 3.10
* Flask
* Machine Learning
* Deep Learning

Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap

Machine Learning

* Autoencoder
* Attention-based LSTM
* SMOTE
* Feature Selection
* Feature Scaling

Dataset

IoT-23 Dataset

The project documentation specifies Python 3.10 and Flask for the simulation/backend environment and HTML, CSS, and JavaScript for the frontend.

📂 Project Structure

```text
Intrusion-Detection/
│
├── app.py
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── login.html
│   ├── register.html
│   ├── predict.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   ├── bootstrap.css
│   │   ├── bootstrap.min.css
│   │   └── style.css
│   │
│   ├── js/
│   │   ├── jquery-3.4.1.min.js
│   │   └── bootstrap.js
│   │
│   └── images/
│
├── model/
│   └── trained_model_files
│
├── dataset/
│   └── IoT-23 dataset files
│
└── README.md
```

> **Note:** Adjust the filenames above according to the actual files in your GitHub repository.

⚙️ Requirements

Before running the project, install:

* Python 3.10 or later
* pip
* Git
* A code editor such as Visual Studio Code

Recommended system requirements from the project documentation include Windows 7 or later, at least 6 GB RAM, an i5 or above processor, and at least 2 GB free disk space.

🚀 How to Run the Application

1. Clone the Repository

Open Command Prompt or Terminal and run:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Move into the project folder:

```bash
cd YOUR-REPOSITORY
```

2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

If you are using PowerShell:

```bash
venv\Scripts\Activate.ps1
```

3. Install Dependencies

If your repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, install the required packages used by your Python application, for example:

```bash
pip install flask numpy pandas scikit-learn imbalanced-learn tensorflow
```

> Install only the packages actually imported by your `app.py` and model files.

4. Check the Project Files

Make sure your Flask application file is present, for example:

```text
app.py
```

Also make sure the required trained model files and supporting files are available in the locations expected by your Python code.

### 5. Start the Flask Application

Run:

```bash
python app.py
```

If your application uses Flask's standard development server, you should see a local address similar to:

```text
http://127.0.0.1:5000/
```

Open this address in your web browser.

🌐 Application Flow

```text
Home Page
    ↓
Register
    ↓
Login
    ↓
Prediction Page
    ↓
Enter Network Traffic Parameters
    ↓
Submit Prediction
    ↓
Machine Learning Model
    ↓
Intrusion Detection Result
```

The prediction page accepts network-flow parameters such as destination port, protocol, flow duration, forward/backward packets, packet lengths, flow bytes/packets per second, and flow inter-arrival-time values.

🔐 User Registration

Users can create an account using:

* Username
* Email
* Password
* Confirm Password

The registration form submits the information to the Flask `/register` route.

🤖 Intrusion Prediction

After logging in, users can provide network traffic parameters through the prediction form.

The application sends the submitted data to:

```text
/predict_page
```

The trained model processes the input and generates an intrusion detection result.

📊 Result

The prediction result is displayed on the result page.

```text
Intrusion Detection Result

[ Prediction Result ]

Go Back
```

The result page receives the prediction through the Flask template variable `result`.

🧠 Model Architecture

```text
IoT-23 Dataset
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Feature Scaling + SMOTE
       ↓
Autoencoder
       ↓
Feature Extraction
       ↓
Attention-Based LSTM
       ↓
Intrusion Detection
       ↓
Prediction Result
```

The proposed architecture combines an Autoencoder for feature extraction/dimensionality reduction with an attention-enhanced LSTM for sequential network-traffic analysis.

📚 Dataset

This project uses the **IoT-23 dataset**, containing network traffic from IoT environments with benign and malicious activities.

The preprocessing pipeline includes:

* Data cleaning
* Data transformation
* Feature scaling
* Feature selection
* SMOTE for class balancing

✨ Features

* Real-time intrusion detection
* Machine Learning-based anomaly detection
* Hybrid Autoencoder–Attention LSTM model
* IoT network security
* User registration and login
* Web-based prediction interface
* Prediction result page
* Flask backend
* Responsive frontend

🔮 Future Enhancements

Possible future improvements include:

* Federated learning
* Adaptive learning for emerging threats
* Lightweight models for resource-constrained IoT devices
* Blockchain-based secure data sharing
* Advanced real-time security dashboards

These enhancements are also identified in the project's documentation.

👨‍💻 Project Contributors

* Abdul Kareem N
* **Balaji V**
* Iswarya S
* Sandhiya A

📄 Project Information

**Project:** Intrusion Detection Using Wireless Network in Machine Learning

**Backend:** Python + Flask

**Frontend:** HTML + CSS + JavaScript

**Dataset:** IoT-23

**Model:** Hybrid Autoencoder–Attention LSTM

⚠️ Disclaimer

This project is developed for educational and research purposes. Prediction performance depends on the trained model, dataset, preprocessing pipeline, and input network-traffic features.

⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
