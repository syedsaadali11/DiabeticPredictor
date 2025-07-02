from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.urls import reverse
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import Booster
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define BASE_DIR correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This points to D:\diabeetic\diabetes_predictor

# Load models and scaler with error handling
try:
    catboost_model = CatBoostClassifier().load_model(os.path.join(BASE_DIR, 'predictor', 'models', 'catboost.cbm'))
except Exception as e:
    print(f"Error loading CatBoost model: {e}")
    catboost_model = None

try:
    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(BASE_DIR, 'predictor', 'models', 'xgboost.json'))
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None

try:
    lgb_model = Booster(model_file=os.path.join(BASE_DIR, 'predictor', 'models', 'lightgbm.txt'))
except Exception as e:
    print(f"Error loading LightGBM model: {e}")
    lgb_model = None

try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'predictor', 'models', 'scaler.pkl'))
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Check if models and scaler are loaded
if catboost_model is None or xgb_model is None or lgb_model is None or scaler is None:
    raise Exception("One or more models/scaler failed to load. Check the file paths and ensure the files exist.")

W_CAT = 0.780
W_XGB = 0.102
W_LGB = 0.118

# NEW calibrated thresholds
THRESH_DIAB = 0.050   # class‑2 (“Diabetes”)
THRESH_PRE  = 0.001   # class‑1 (“Pre‑Diabetes”)

QUESTIONS = [
    {"name": "HighBP", "text": "Have you been told you have high blood pressure?", "type": "yesno"},
    {"name": "HighChol", "text": "Have you been told you have high cholesterol?", "type": "yesno"},
    {"name": "CholCheck", "text": "Have you had your cholesterol checked in the last 5 years?", "type": "yesno"},
    {"name": "BMI", "text": "What is your Body Mass Index (BMI)? Example: 25 = normal, 30+ = overweight", "type": "number"},
    {"name": "Smoker", "text": "Have you smoked 100 cigarettes or more in your life?", "type": "yesno"},
    {"name": "Stroke", "text": "Have you ever had a stroke?", "type": "yesno"},
    {"name": "HeartDiseaseorAttack", "text": "Do you have heart disease or have had a heart attack?", "type": "yesno"},
    {"name": "PhysActivity", "text": "Do you engage in regular physical activity?", "type": "yesno"},
    {"name": "Fruits", "text": "Do you eat fruit at least once per day?", "type": "yesno"},
    {"name": "Veggies", "text": "Do you eat vegetables at least once per day?", "type": "yesno"},
    {"name": "HvyAlcoholConsump", "text": "Do you drink heavily? (Men: 14+ drinks/week, Women: 7+)", "type": "yesno"},
    {"name": "AnyHealthcare", "text": "Do you currently have any form of healthcare coverage?", "type": "yesno"},
    {"name": "NoDocbcCost", "text": "Have you skipped doctor visits due to cost in the last year?", "type": "yesno"},
    {"name": "GenHlth", "text": "How would you rate your overall health? (1 = Excellent, 5 = Poor)", "type": "scale", "range": (1, 5)},
    {"name": "MentHlth", "text": "In the past 30 days, how many days was your mental health not good? (0–30)", "type": "scale", "range": (0, 30)},
    {"name": "PhysHlth", "text": "In the past 30 days, how many days was your physical health not good? (0–30)", "type": "scale", "range": (0, 30)},
    {"name": "DiffWalk", "text": "Do you have serious difficulty walking or climbing stairs?", "type": "yesno"},
    {"name": "Sex", "text": "Are you male?", "type": "yesno"},  # 1 = Male, 0 = Female
    {"name": "Age", "text": "What is your age group? (1=1-15, 2=16-30, 3=31-45, 4=46-60, 5=61+)", "type": "scale", "range": (1, 5)},
    {"name": "Education", "text": "What is your highest education level? (1=No School, 6=College Graduate)", "type": "scale", "range": (1, 6)},
    {"name": "Income", "text": "What is your household monthly income in PKR? (1=Below 30k, 8=Above 150k)", "type": "scale", "range": (1, 8)},
]

def home(request):
    section = request.GET.get('section', 'landing')  # Default to landing

    # Clear session data if the user is not authenticated or navigating to a non-question section
    if not request.user.is_authenticated or section in ['landing', 'home', 'intro', 'result']:
        if 'step' in request.session:
            del request.session['step']
        if 'answers' in request.session:
            del request.session['answers']
        request.session.modified = True

    # Handle signup
    if request.method == 'POST' and 'signup' in request.POST:
        form = UserCreationForm(request.POST)
        username = request.POST.get('username')
        if User.objects.filter(username=username).exists():
            context = {
                'signup_form': form,
                'login_form': AuthenticationForm(),
                'signup_error': "This username is already registered. Please try a different username or log in."
            }
            return render(request, 'predictor/landing.html', context)
        if form.is_valid():
            form.save()
            user = authenticate(username=form.cleaned_data['username'], password=form.cleaned_data['password1'])
            login(request, user)
            request.session['signup_success'] = "Successfully registered! Now you can ready to use our services."
            return redirect(reverse('home') + '?section=landing')
    signup_form = UserCreationForm()

    # Handle login
    if request.method == 'POST' and 'login' in request.POST:
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect(reverse('home') + '?section=home')
        else:
            context = {
                'signup_form': UserCreationForm(),
                'login_form': form,
                'login_error': "Invalid username or password. Please try again."
            }
            return render(request, 'predictor/landing.html', context)
    login_form = AuthenticationForm()

    # Handle predictor
    if request.method == 'POST' and 'predict' in request.POST:
        logger.debug(f"POST request received. Current step: {request.session.get('step')}")
        if 'step' not in request.session:
            logger.debug("Initializing step to 0")
            request.session['step'] = 0
            request.session['answers'] = {}

        step = request.session['step']
        logger.debug(f"Processing step: {step}")
        if step < len(QUESTIONS):
            q = QUESTIONS[step]
            answer = request.POST.get(q['name'], '')
            logger.debug(f"Question: {q['name']}, Answer: {answer}")
            if q['type'] == 'yesno':
                if answer.lower() not in ['yes', 'no']:
                    context = {
                        'question': q['text'],
                        'name': q['name'],
                        'type': q['type'],
                        'range': q.get('range', None),
                        'step': step + 1,
                        'total': len(QUESTIONS),
                        'error_message': "Please select either 'Yes' or 'No'."
                    }
                    return render(request, 'predictor/question.html', context)
                answer = 1 if answer.lower() == 'yes' else 0
            else:  # For number or scale questions
                if not answer or answer.lower() == 'no':
                    context = {
                        'question': q['text'],
                        'name': q['name'],
                        'type': q['type'],
                        'range': q.get('range', None),
                        'step': step + 1,
                        'total': len(QUESTIONS),
                        'error_message': f"Please enter a valid number for {q['name']}."
                    }
                    return render(request, 'predictor/question.html', context)
                try:
                    answer = float(answer)
                    if q['type'] == 'scale':
                        min_val, max_val = q['range']
                        if not (min_val <= answer <= max_val):
                            context = {
                                'question': q['text'],
                                'name': q['name'],
                                'type': q['type'],
                                'range': q.get('range', None),
                                'step': step + 1,
                                'total': len(QUESTIONS),
                                'error_message': f"Please enter a number between {min_val} and {max_val} for {q['name']}."
                            }
                            return render(request, 'predictor/question.html', context)
                except ValueError:
                    context = {
                        'question': q['text'],
                        'name': q['name'],
                        'type': q['type'],
                        'range': q.get('range', None),
                        'step': step + 1,
                        'total': len(QUESTIONS),
                        'error_message': f"Please enter a valid number for {q['name']}."
                    }
                    return render(request, 'predictor/question.html', context)
            request.session['answers'][q['name']] = answer
            step += 1
            request.session['step'] = step
            request.session.modified = True
            logger.debug(f"Step incremented to: {step}, Session step: {request.session['step']}")
            # Redirect to the same view to ensure the next question is loaded
            return redirect(reverse('home') + '?section=question')

        if step >= len(QUESTIONS):
            features = [request.session['answers'].get(q['name'], 0) for q in QUESTIONS]
            input_data = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            # Get prediction probabilities
            cat_proba = catboost_model.predict_proba(input_scaled)
            xgb_proba = xgb_model.predict_proba(input_scaled)
            lgb_proba = lgb_model.predict(input_scaled, raw_score=False)
            ensemble_proba = W_CAT * cat_proba + W_XGB * xgb_proba + W_LGB * lgb_proba
            p0, p1, p2 = ensemble_proba[0]          # unpack into three scalars

            if p2 >= THRESH_DIAB:
               prediction = 2                      # Diabetes
            elif p1 >= THRESH_PRE:
               prediction = 1                      # Pre‑Diabetes
            else:
               prediction = 0                      # No Diabetes

            result = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}[prediction]


            # Store only the result in the session
            request.session['result'] = result
            return redirect(reverse('home') + '?section=result&show_result=true')

    # Handle contact form (placeholder)
    if request.method == 'POST' and 'contact' in request.POST:
        return redirect(reverse('home') + '?section=contact')

    # Prepare context
    context = {
        'signup_form': signup_form,
        'login_form': login_form,
    }

    # Add signup success message if present
    if 'signup_success' in request.session:
        context['signup_success'] = request.session['signup_success']
        del request.session['signup_success']

    # Add question context if in question section
    if request.user.is_authenticated and 'step' in request.session and request.session['step'] < len(QUESTIONS):
        current_question = QUESTIONS[request.session['step']]
        context.update({
            'question': current_question['text'],
            'name': current_question['name'],
            'type': current_question['type'],
            'range': current_question.get('range', None),
            'step': request.session['step'] + 1,
            'total': len(QUESTIONS),
        })
        section = 'question'
        logger.debug(f"Rendering question page. Step: {request.session['step']}, Question: {current_question['text']}")

    # Add result context if in result section
    elif section == 'result' and 'result' in request.session and request.GET.get('show_result'):
        context['result'] = request.session['result']
        request.session.flush()

    return render(request, 'predictor/question.html' if section == 'question' else 'predictor/landing.html' if section == 'landing' and not request.user.is_authenticated else 'predictor/home.html' if section == 'home' else 'predictor/intro.html' if section == 'intro' else 'predictor/result.html' if section == 'result' else 'predictor/landing.html', context)

def user_logout(request):
    logout(request)
    # Clear the entire session on logout
    request.session.flush()
    return redirect(reverse('home') + '?section=landing')
