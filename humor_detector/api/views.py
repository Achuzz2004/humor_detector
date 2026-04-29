import os
import pickle
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.conf import settings 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Use absolute paths relative to your project root
# This assumes 'models' is a folder inside your humor_detector project folder
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', 'Hybrid_CNN_LSTM.keras')
TOKENIZER_PATH = os.path.join(settings.BASE_DIR, 'models', 'tokenizer.pkl')

# Global variable to hold the model in memory
# Wrap in a try-except if you want to avoid crashes during initial setup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    model = None
    tokenizer = None


def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
        # This loop removes the help_text from all fields (Username, Password, etc.)
        for field in form.fields.values():
            field.help_text = None 
            
    return render(request, 'register.html', {'form': form})

def login_view(request):
    """
    Standard Login using Django's built-in AuthenticationForm.
    """
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """
    Logs out user and aggressively clears cache headers to block backspace access.
    """
    logout(request)
    response = redirect('login')
    
    # Security: Explicitly tell the browser to kill all cache for this session
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    
    return response



@login_required(login_url='login')
@never_cache
def humor_classifier(request):
    result = None
    
    if request.method == "POST":
        text = request.POST.get('text_input', '').strip()
        
        if text and model and tokenizer:
            # 1. Preprocess
            sequences = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=60, padding='post', truncating='post')

            # 2. Predict
            logits = model.predict(padded)
            # Use tf.sigmoid because your training used 'from_logits=True'
            prob = float(tf.sigmoid(logits).numpy()[0][0])
            
            # 3. Format result
            is_humorous = prob > 0.5
            result = {
                "text": text,
                "label": "Humorous" if is_humorous else "Not Humorous",
                "confidence": f"{round(prob * 100 if is_humorous else (1-prob) * 100, 2)}%",
                "is_humorous": is_humorous
            }
        elif not text:
             result = {"error": "Please enter some text!"}

    return render(request, 'index.html', {'result': result})

