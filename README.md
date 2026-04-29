# Humor Detector

A Django-based web application for humor detection using a Hybrid CNN-LSTM deep learning model.

## Features

- User registration and authentication
- Humor classification API
- Web interface for text classification
- Pre-trained Hybrid CNN-LSTM model

## Prerequisites

- Python 3.12.0 or higher
- Virtual environment (recommended)

## Installation

1. **Clone or download the project**

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to the Django project:**
   ```bash
   cd humor_detector
   ```

5. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```


## Running the Application

1. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:8000/
   ```

## Usage

- **Register:** Create a new account at `/register/`
- **Login:** Sign in at `/login/`
- **Classify Text:** Use the main interface to input text and get humor probability

## Model Details

- **Architecture:** Hybrid CNN-LSTM with multi-kernel convolution (filter sizes: 3, 5, 7)
- **Input:** Text sequences (max length: 60 tokens)
- **Vocabulary:** 25,000 words
- **Output:** Probability score (0-1) indicating humor likelihood

## Training

The model was trained using the notebook `FSECE2026_68.ipynb` with:
- Dataset: 200k Short Texts for Humor Detection (Kaggle)
- Training split: 72% train, 8% validation, 20% test
- Epochs: 15
- Batch size: 64

## Project Structure

```
humor_detector/
├── api/                    # Main Django app
│   ├── templates/         # HTML templates
│   ├── models.py          # Database models
│   ├── views.py           # View logic and ML inference
│   └── urls.py            # URL routing
├── humor_detector/        # Django project settings
├── models/                # ML model artifacts
│   ├── Hybrid_CNN_LSTM.keras
│   └── tokenizer.pkl
├── db.sqlite3             # SQLite database
└── manage.py
```

## API Endpoints

- `GET /` - Home page (requires login)
- `GET /register/` - User registration
- `GET /login/` - User login
- `POST /logout/` - User logout

## Technologies Used

- Django 6.0.3
- TensorFlow 2.21.0
- Keras 3.14.0
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn