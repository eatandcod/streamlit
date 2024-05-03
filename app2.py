import time
from inference_sdk import InferenceHTTPClient
import streamlit as st
import sqlite3
import hashlib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import cv2
import tensorflow as tf
import io
import base64

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@st.cache_resource
def load_my_model():
    model = load_model('fire_detection_model.h5')
    return model


@st.cache_resource
def load_rf_model():
    clf = joblib.load('wildfire_random_forest_model.pkl')
    return clf


def load_svm():
    clf = joblib.load('svm_fire.pkl')
    return clf


# Load your model
model = load_my_model()
rf_model = load_rf_model()
svm_mod = load_svm()
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="MGKOlmaEql2wSPqyWeOU"  # Replace with your actual Roboflow API key
)
def save_image(captured_image):
    # Save the PIL image to a temporary file and return the file path
    if captured_image is not None:
        image = Image.open(io.BytesIO(captured_image.getvalue()))
        image_path = "temp_image.jpg"  # Temporary file name
        image.save(image_path, "JPEG")  # Save image as JPEG
        return image_path
    return None
def perform_inference(image_path, model_id):
    # Use the InferenceHTTPClient to perform inference using the image file path
    result = CLIENT.infer(image_path, model_id=model_id)
    return result

# Function to preprocess and predict the fire in the image
def predict(model, img):
    img = img.resize((64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    # Use the model to predict
    result = model.predict(img)
    # st.write('Debug: Raw prediction output:', result)  # Debug information
    return 'fire' if result[0][0] < 0.5 else 'no fire'


def predict_rf(model, img):
    img = img.resize((64, 64))
    img = np.array(img)
    img = img.flatten()  # Flatten the image
    img = img.reshape(1, -1)  # Reshape for the model
    result = model.predict(img)
    return 'fire' if result[0] == 1 else 'no fire'


svm_model = joblib.load('svm_fire.pkl')


# Function to predict using SVM
def predict_svm(image):
    image = cv2.resize(image, (64, 64))  # Resize image
    image = image.flatten().reshape(1, -1)  # Flatten and reshape image
    prediction = svm_model.predict(image)  # Predict using the SVM model
    return 'fire' if prediction[0] == 1 else 'no fire'


# Rothermel spread rate calculation function
def calculate_spread_rate(I_R, rho_b, epsilon, Q_ig, phi_W, phi_S):
    heat_source = I_R * (1 + phi_W + phi_S)
    heat_sink = rho_b * epsilon * Q_ig
    R = heat_source / heat_sink
    return R


# Function to plot the graph for Rothermel spread rate
def plot_spread_rate(phi_range, spread_rates, factor_name):
    plt.figure(figsize=(10, 4))
    plt.plot(phi_range, spread_rates, '-o')
    plt.title(f'Fire Spread Rate vs {factor_name}')
    plt.xlabel(f'{factor_name} Factor')
    plt.ylabel('Fire Spread Rate (m/min)')
    plt.grid(True)
    st.pyplot(plt)


# Navigation Sidebar


def audio_alert():
    # Open audio file
    audio_file = open('sound/alert_sound.mp3', 'rb')
    audio_bytes = audio_file.read()

    # Convert audio bytes to base64
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

    # Embed as HTML with autoplay attribute set to true
    audio_html = f"""
    <audio autoplay controls style="display:none;">
      <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
    Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


from streamlit_elements import elements, mui, sync


# Define your app logic
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Set the background to a specified image file (e.g., 'background.png')
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Call the function to set the background
set_background('static/img_3.png')

import base64


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


image_base64 = get_base64_of_image("static/img_3.png")


def verify_login(username, password):
    conn = create_connection()
    hashed_password = hash_password(password)
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_password))
            user = c.fetchone()
            conn.close()
            return user
        except sqlite3.Error as e:
            st.error(e)
            return None


# Function to show the login form
st.markdown("""
    <style>
    /* Targeting all Streamlit buttons */
    button {
        color: orange !important; /* Setting the text color to orange */
        background-color: brown !important; /* Setting the background color to brown */
        border-radius: 5px !important; /* Adding rounded corners to the buttons */
        border: 1px solid orange !important; /* Adding an orange border for consistency */
        padding: 10px 20px !important; /* Adjusting padding to increase button size */
        font-weight: bold !important; /* Making the button text bold */
    }
    /* Adjusting hover effect */
    button:hover {
        background-color: #8B4513 !important; /* Making button background darker on hover */
    }
    </style>
""", unsafe_allow_html=True)
def show_login_page():
    if st.button('Forgot Password'):
        # This will switch the page to the "Forgot Password" page
        st.session_state['page'] = 'forgot_password'
    # Set the background image for the login page
    set_background('static/img_6.png')  # Assuming img_3.png is the background you want for the login page

    # Include your custom styles here (similar to the ones in show_create_account)
    # Custom styles, similar to those used in the create account page
    st.markdown("""
     <style>
     #Login .stTextInput input, #Login .stPassword input {
    color: #4F4F4F;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    padding: 10px;
}
.stButton > button {
    color: #fff;
    background-color: #FF6347;
    border-radius: 20px;
    border: 2px solid #000000; /* Dark black border */
    padding: 10px 24px;
    margin-top: 10px;
    margin-bottom: 10px; /* Add space below the button if needed */
    box-shadow: none;
    font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
}

.stButton > button:hover {
    background-color: #FF4500;
    border-color: #000000; /* Keep the black border on hover */
}


#Login .stButton > button:hover {
    background-color: #FF4500;
}

/* If there's a need to adjust the form container specifically for the login, use this */
#Login .stForm {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 2rem;
    margin-top: 3rem;
    border: 2px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
} 
h2 {
        font-size: 2.5em; /* Larger font size */
        color: #ffffff; /* White color for better contrast */
        text-shadow: 2px 2px 4px #000000; /* Text shadow for readability */
        text-align: center; /* Center the text */
        margin-bottom: 1rem; /* Space below the title */
    }

    /* Style adjustments for the form container */
    .stForm {
        background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
        border-radius: 15px; /* Rounded corners */
        padding: 2rem; /* Padding around the form */
        margin-top: 3rem; /* Space from the top */
        border: 2px solid rgba(255, 255, 255, 0.3); /* Subtle border */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Light shadow for depth */
    }

    /* Style for the 'Create Account' link */
    .stButton > a {
        font-size: 1.1em; /* Larger font size */
        color: #FF6347; /* Warm red text color to match the button */
        text-decoration: none; /* No underline */
    }

    .stButton > a:hover {
        color: #FF4500; /* Darker orange color on hover */
    }

    /* Style for the form inputs and labels */
    .stTextInput > label, .stPassword > label, .stTextInput input, .stPassword input {
        /* Your existing styles */
    }

    /* Style for the login button */
    .stButton > button {
        /* Your existing styles */
    }

    /* Adjustments for the 'Don't have an account?' text */
    .stMarkdown {
        font-size: 1em; /* Adjust size as needed */
        color: #ffffff; /* White color for better contrast */
        text-align: center; /* Center the text */
        margin-top: 1rem; /* Space above the text */
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("Login", clear_on_submit=True):
        st.markdown("## Login to Your Account")
        username = st.text_input('Username', placeholder="Username")
        password = st.text_input('Password', type='password', placeholder="Password")
        submit_button = st.form_submit_button(label='Login')

        if submit_button:
            user = verify_login(username, password)
            if user:
                st.session_state['logged_in'] = True
                st.success('Logged in successfully!')
                st.experimental_rerun()
            else:
                st.error('Invalid username or password.')

    # Add a link for users to create a new account if they don't have one
    st.markdown("## Don't have an account?")


def recover_password():
    st.markdown("Forgot your password? Contact [support@example.com](mailto:support@example.com) for help.")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to create a connection to the SQLite database
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('identifier.sqlite')  # Path to the SQLite database file
    except sqlite3.Error as e:
        st.error(e)
    return conn


# Function to create the Users table if it doesn't exist
def create_users_table():
    conn = create_connection()
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            st.error(e)
        finally:
            conn.close()


create_users_table()  # Call this function to ensure the table is created


# Function to insert a new user into the database
def insert_new_user(username, password, email):
    conn = create_connection()
    hashed_password = hash_password(password)
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                      (username, hashed_password, email))
            conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            st.error(f"Username or email already exists. Please choose another one. Error: {e}")
            return False
        finally:
            conn.close()


# Check if the user is already logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover; 
        background-position: top -300px center; /* Adjust the position to move the background higher */
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Style for the form inputs and labels */
    .stTextInput > label, .stPassword > label, .stEmail > label, .stTextInput input, .stPassword input, .stEmail input {{
        color: #FF6347; /* Warm red text color that matches the button */ 
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif; /* A modern sans-serif font */
    }}

    /* Style for the button */
    .stButton > button {{
        color: #fff; /* White text color */
        background-color: #FF6347; /* Warm red background color */
        border-radius: 20px; /* Rounded corners */
        border: none; /* No border */
        padding: 10px 24px; /* Padding inside the button */
        margin-top: 10px; /* Margin above the button */
        box-shadow: none; /* No shadow */
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif; /* Matching font for consistency */
    }}

    /* Change the hover effect color for the button */
    .stButton > button:hover {{
        background-color: #FF4500; /* A slightly darker orange color on hover */
    }}

    /* Style adjustments for the form container */
    .stForm {{
        background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
        border-radius: 15px; /* Rounded corners */
        padding: 2rem; /* Padding around the form */
        margin-top: 3rem; /* Space from the top */
        border: 2px solid rgba(255, 255, 255, 0.3); /* Subtle border */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Light shadow for depth */
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


import streamlit as st


def show_create_account():
    # Set the background image for the create account page
    set_background('static/img_6.png')
    st.markdown("""
       <style>
       /* Style for the form */



       /* Custom styles for the input fields */
       .stTextInput > div > div > input {
           color: #4F4F4F; /* Dark grey color for the text */
    background-color: rgba(255, 255, 255, 0.9); /* Slightly transparent white for the input background */
    border-radius: 20px; /* Rounded corners for the input fields */
    border: 1px solid rgba(0, 0, 0, 0.1); /* Subtle border color */
       }

       /* Custom styles for the label */
       .stForm label {
    color: #E8E8E8; /* Light grey color for the text */
    font-family: 'Arial', sans-serif; /* Modern font */
} 
.stTextInput input, .stPassword input, .stEmail input {
    color: #2E2E2E; /* Darker text color for better visibility */
    background-color: #FFFFFF; /* White background for the input */
    border-radius: 20px; /* Rounded corners for the input fields */
    border: 1px solid #CCCCCC; /* Light grey border */
    padding: 10px; /* Padding inside the input fields */
}

       /* Style for the button */
       .stButton > button {
           color: #fff; /* White color for the button text */
        background-color: #FF6347; /* Tomato color for the button background */
        border-radius: 20px; /* Rounded corners for the button */
        border: none; /* No border for the button */
        padding: 10px 24px; /* Padding inside the button */
        margin-top: 10px; /* Margin above the button */
        box-shadow: none; /* No shadow for the button */
       } 
       .stButton > button:hover {
        background-color: #FF4500; /* Darker orange color on hover */
    }

       .stButton > button:hover {
           background-color: #0056a3; /* Darker blue background on hover */
       } 
       .stTextInput > div > div > input, .stPassword > div > div > input, .stEmail > div > div > input {
        font-size: 18px; /* Larger font size for the input text */
        height: 50px; /* Taller input fields for a larger clickable area */
        background-color: #ffffff; /* White background for the input */
        color: #333333; /* Dark grey color for the text, ensures better visibility */
        border-radius: 25px; /* Rounded corners for the input fields */
        border: 2px solid #FFA07A; /* Border color to match your theme */
        padding: 0 15px; /* Padding inside the input fields */
    }

    /* Custom styles for the input labels */
    .stForm > .stForm > .stForm > .stForm > .stForm > label {
        font-size: 700px; /* Larger font size for the labels */
        color: #FFA07A; /* Color for the labels to match your theme */
    }

       /* Style for the form container */
       .stForm {
           background-color: rgba(255, 255, 255, 0.8); /* Light background with some opacity */
    border-radius: 15px; /* Rounded corners */
    padding: 2rem; /* Padding around the form */
    margin-top: 3rem; /* Space from the top */
    border: 2px solid rgba(255, 255, 255, 0.3); /* Slightly visible border */
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
       }
       </style>
       """, unsafe_allow_html=True)
    # Title styling

    with st.form('Create Account'):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        email = st.text_input('Email')
        submit_button = st.form_submit_button(label='Create Account')

        if submit_button:
            if insert_new_user(username, password, email):
                st.session_state['logged_in'] = True
                st.success('Account created successfully!')
                st.experimental_rerun()  # Rerun the app which will now show the dashboard
            else:
                st.error('Error creating your account.')


# Main dashboard page
def show_dashboard():
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select a page:', ['Home', 'Fire Images', 'News', 'Contact', 'R Shiny APP'])
    st.markdown(f"""
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{image_base64}");
    background-size: cover !important; 
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;

    }}  
    /* This targets the file uploader button by its role attribute */
    div[role="button"][aria-label="Upload"] {{
        background-color: #66BB6A !important;
    }}

    /* This styles the main title text */
    .header-title {{
        font-size: 3em; /* Large font size */
        font-weight: bold; /* Bold font */
        color: #013220; /* Light text color for contrast */
        text-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); /* Text shadow for depth */
        margin-bottom: 0.5em; /* Space below the title */
    }}

    /* Apply the custom style to the title */
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap'); /* Importing the 'Lobster' font from Google Fonts */

        /* Custom CSS for the title */
        .title {

            text-align: center; /* Center the title */
            margin-top: 0.5em; /* Adjust the top margin to position the title */
        } 

        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Custom CSS for moving the text to the left */
        .move-left {
            position: relative;
            right: 50%; /* Adjust the percentage to move the text further to the left */ 
            border: 2px solid red;
        border-radius: 4px;
        padding: 8px;
        box-shadow: 0 0 8px red;
        display: inline-block;
        color: white;
        background-color: black;
        margin: 10px; 
        font-family: 'Lobster', cursive; /* Using the 'Lobster' font */
            font-size: 2em; /* Adjust the size as needed */
            color: #FFA07A; /* Choose a color that stands out on your background */
            text-shadow: 2px 2px 4px #000000; /* Shadow effect for depth */ 
            animation-name: pulseAnimation;
            animation-duration: 2s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out; 

        } 
        /* Keyframes for animation */
        @keyframes pulseAnimation {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        /* Custom CSS for the animated text */
        .animated-text {
            animation-name: pulseAnimation;
            animation-duration: 2s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use the custom class for your text
    st.markdown('<div class="move-left">Welcome to the Forest Fire Detection Dashboard</div>', unsafe_allow_html=True)

    # Home Page with Image Prediction
    if options == 'Home':

        # Define custom styles for buttons
        button_style = """
        <style>
        .custom-button {
            background-color: #FFA07A; /* Light shade of orange */
            border: none;
            color: black;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px; /* Rounded corners */
        }

        .custom-button:hover {
            background-color: #ff7f50; /* Slightly darker shade of orange */
        }
        .header-title {
        font-size: 3em; /* Large font size */
        font-weight: bold; /* Bold font */
        color: #FFFFFF; /* White text color for contrast */
        text-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); /* Text shadow for depth */
        margin-bottom: 0.5em; /* Space below the title */
    }
        .button-container {
            text-align: left; /* Align the buttons to the left */
        }
        </style>
        """
        # Apply the custom styles
        st.markdown(button_style, unsafe_allow_html=True)

        # Create button container
        st.markdown('<div class="button-container">', unsafe_allow_html=True)

        # Upload Image Button
        st.markdown(
            '<button class="custom-button" onclick="document.getElementById(\'uploadButton\').click()">Upload Image</button>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key="uploadButton",
                                         help="Upload your image here")

        # Capture Image Button
        st.markdown(
            '<button class="custom-button" onclick="document.getElementById(\'captureButton\').click()">Capture Image</button>',
            unsafe_allow_html=True
        )
        captured_image = st.camera_input("", key="captureButton", help="Capture your image here")
        if captured_image is not None and st.button('Detect Fire'):
            image_path = save_image(captured_image)

            if image_path:
                model_id = "forest-fire-detection-ilo2d/1"  # Replace with your actual model ID

                try:
                    results = perform_inference(image_path, model_id)
                    st.write("Detection Results:")

                    # Check if the predictions list is empty
                    if not results["predictions"]:  # This checks if predictions list is empty
                        st.write("No Fire Detected.")  # Display when no fire is detected
                    else:
                        st.json(results)  # Display the raw JSON if predictions are present

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)
        st.markdown('</div>', unsafe_allow_html=True)  # Close button container

        # If an image was uploaded or captured, display and predict
        if uploaded_file is not None or captured_image is not None:
            img = Image.open(uploaded_file) if uploaded_file is not None else Image.open(
                io.BytesIO(captured_image.getvalue()))
            st.image(img, caption='Selected Image', use_column_width=True)

            # Prediction logic goes here...

            def show_prediction_result(prediction):
                # Define the CSS styles
                result_style = """
                <style>
                    .prediction-box {
                        border: 2px solid #013220; /* Dark green border */
                        background-color: #A5D6A7; /* Light green background */
                        border-radius: 5px; /* Rounded corners */
                        padding: 10px; /* Padding inside the box */
                        margin: 10px 0; /* Margin outside the box */
                        color: #013220; /* Dark green text */
                    }
                </style>
                """

                # Use markdown to display the prediction result with the defined style
                st.markdown(result_style, unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-box'>{prediction}</div>", unsafe_allow_html=True)

            # CNN Prediction Button
            if st.button('Predict Image with CNN'):
                with st.spinner('Predicting with CNN...'):
                    prediction = predict(model, img)
                    show_prediction_result(f'The CNN prediction is: {prediction}')
                    if prediction == 'fire':
                        audio_alert()  # Play the alert sound if fire is detected

            # Random Forest Prediction Button
            if st.button('Predict Image with Random Forest'):
                with st.spinner('Predicting with Random Forest...'):
                    prediction = predict_rf(rf_model, img)
                    show_prediction_result(f'The RF prediction is: {prediction}')
                    if prediction == 'fire':
                        audio_alert()  # Play the alert sound if fire is detected

            # SVM Prediction Button
            if st.button('Predict Image with SVM'):
                with st.spinner('Predicting with SVM...'):
                    # Initialize variable for the image array
                    opencv_image = None

                    # Check if an uploaded image is available
                    if uploaded_file is not None:
                        img = Image.open(uploaded_file).convert('RGB')
                        opencv_image = np.array(img)  # Convert PIL image to a numpy array

                    # Check if an image was captured with the camera
                    elif captured_image is not None:
                        # Convert the captured image to a numpy array
                        img = Image.open(io.BytesIO(captured_image.getvalue()))
                        opencv_image = np.array(img)  # Convert PIL image to a numpy array

                    # If an image was uploaded or captured, make a prediction
                    if opencv_image is not None:
                        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                        prediction = predict_svm(opencv_image)  # Make prediction
                        show_prediction_result(f'The svm prediction is: {prediction}')
                        if prediction == 'fire':
                            audio_alert()

    # Spread Rate Page
    elif options == 'R Shiny APP':
        st.markdown("""
            <style>
                .shiny-app-link-button {
                    background-color: #FFA500; /* Orange background */
                    border: none;
                    color: #654321; /* Dark brown text color */
                    padding: 15px 30px; /* Padding */
                    font-size: 20px; /* Font size */
                    text-align: center;
                    text-decoration: none;
                    display: block; /* Change to block for full width, helps to align to the left */
                    cursor: pointer;
                    border-radius: 5px; /* Rounded corners */
                    box-shadow: 2px 2px 4px #FF8C00; /* Orange shadow */
                    font-weight: bold;
                    margin: 10px 0px; /* Top and bottom margin */
                    animation: pulse 2s infinite ease-in-out;
                    text-align: left; /* Aligns the text to the left */
                    margin-left: 50px; /* Aligns the button to the left */
                    margin-right: auto; /* Centers the button horizontally if needed */
                    max-width: max-content; /* Maximum content width */
                }

                .shiny-app-link-button:hover {
                    background-color: #E68A00; /* Slightly darker orange on hover */
                    box-shadow: 2px 2px 6px #FFA500; /* Slightly larger orange shadow on hover */
                }

                /* Pulse animation */
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }

                /* Position the button below the main title */
                .title-and-button-container {
                    display: block;
                    text-align: center;
                    margin-top: 50px; /* Adjust this to move closer or farther from the title */
                }
            </style>

            <div class='title-and-button-container'>
                <a class='shiny-app-link-button' href='https://finalyear.shinyapps.io/merge/' target='_blank'>
                    Access the R Shiny App here
                </a>
            </div>
        """, unsafe_allow_html=True)

    # ... [your existing imports and code above]

    elif options == 'Fire Images':
        # Title with custom style
        st.markdown("""
                <style>
                    .facts-title {
                        color: #FFA07A; /* Light shade of orange */
                        font-size: 2.5rem; /* Large font size */
                        font-weight: bold; /* Bold font weight */
                        text-shadow: 2px 2px #8B4513; /* Shadow effect */
                        margin-bottom: 1rem; /* Bottom margin */
                        text-align: center; /* Center text alignment */
                        border-bottom: 3px solid #8B4513; /* Underline effect */
                        display: inline-block; /* Block level element */
                        padding-bottom: 0.25rem; /* Padding to the bottom */
                    }
                    .carousel-button {
                        color: #FFA07A; /* Orange text color */
                        background-color: brown; /* Brown background color */
                        padding: 10px 20px; 
                        border-radius: 5px;
                        border: 2px solid #FFA07A;
                        margin: 10px;
                        font-weight: bold;
                    }
                    .carousel-button:hover {
                        background-color: #8B4513; /* Darker brown on hover */
                    }
                    .carousel-container {
                        display: flex;
                        justify-content: space-between; 
                    
                    }
                    .image-caption {
                        background-color: brown; /* Brown background for caption */
                        color: #FFA07A; /* Orange text for caption */
                        padding: 5px;
                        border-radius: 5px;
                        display: inline-block;
                        margin-top: 5px;
                    }
                </style>
                """, unsafe_allow_html=True)

        st.markdown('<h2 class="facts-title">Gallery of the Most Destructive Forest Fires</h2>', unsafe_allow_html=True)

        # Example list of image paths or URLs
        image_list_with_captions = [
            {'image': 'static/img_7.png', 'caption': '2003 Siberian Taiga Fires (Russia) – 55 Million Acres'},
            {'image': 'static/img_8.png', 'caption': '2014 Northwest Territories Fires (Canada) – 8.5 Million Acres'},
            {'image': 'static/img_9.png', 'caption': '2004 Alaska Fire Season (US) – 6.6 Million Acres'},
            {'image': 'static/img_10.png', 'caption': '1939 Black Friday Bushfire (Australia) – 5 Million Acres'},
            {'image': 'static/img_11.png', 'caption': '1919/2020 Australian Bushfires (Australia) – 42 Million Acres'},
            # Add more images with their captions as needed
        ]

        # Initialize the current image index
        if 'current_image' not in st.session_state:
            st.session_state['current_image'] = 0

        # Display navigation buttons and the current image with its caption
        image_placeholder = st.empty()
        caption_placeholder = st.empty()

        # Function to display the image and caption
        def display_image():
            current_image_data = image_list_with_captions[st.session_state['current_image']]
            image_placeholder.image(current_image_data['image'], use_column_width=True)
            caption_placeholder.markdown(f"""
                <style>
                    .custom-caption {{
                        background-color: brown; /* Brown background */
                        color: orange; /* Orange text */
                        font-size: 18px; /* Larger font size */
                        font-weight: bold; /* Bold font weight */
                        padding: 10px; /* Padding inside the box */
                        border-radius: 5px; /* Rounded corners */
                        border: 2px solid orange; /* Orange border */
                        text-align: center; /* Centered text */
                        margin: 10px 0; /* Margin above and below the caption */
                        display: block; /* Use the full width */
                    }}
                </style>
                <div class="custom-caption">{current_image_data['caption']}</div>
            """, unsafe_allow_html=True)

        # Initially display the first image and caption
        display_image()

        # Loop to update the image every 6 seconds
        while True:
            time.sleep(6)  # Wait for 6 seconds
            st.session_state['current_image'] = (st.session_state['current_image'] + 1) % len(image_list_with_captions)
            display_image()
        # Display the caption with custom styling


    elif options == 'News':
        st.markdown("""
                <style>
                    .facts-title {
                        color: #FFA07A; /* Light shade of orange */
                        font-size: 2.5rem; /* Large font size */
                        font-weight: bold; /* Bold font weight */
                        text-shadow: 2px 2px #8B4513; /* Shadow effect */
                        margin-bottom: 1rem; /* Bottom margin */
                        text-align: center; /* Center text alignment */
                        border-bottom: 3px solid #8B4513; /* Underline effect */
                        display: inline-block; /* Block level element */
                        padding-bottom: 0.25rem; /* Padding to the bottom */
                    }
                </style>
                <h2 class="facts-title">Fascinating Facts About Forest Fires</h2>
            """, unsafe_allow_html=True)

        # List of fascinating facts about forest fires
        facts = [
            "Forest fires can travel up to 14 miles per hour, which can outpace humans in a race.",
            "Some fires burn themselves out when they reach a natural firebreak, such as a river.",
            "Lightning strikes are a major natural cause of forest fires.",
            "The smoke from a forest fire can be so thick it creates its own type of weather.",
            "A forest fire's temperature can reach 1,500 degrees Fahrenheit or more.",
            # Add more facts as desired
        ]

        # Function to display a fact in a styled box
        def display_fact(fact):
            # Define CSS to style the fact box
            fact_style = f"""
                    <style>
                        .fact-box {{
                            background-color: #A52A2A; /* Brown background */
                            color: orange; /* Orange text */
                            border-radius: 10px;
                            padding: 20px;
                            margin: 10px 0px;
                            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.5); /* Box shadow */
                            font-weight: bold;
                            text-align: center; /* Centered text */
                        }}
                    </style>
                    <div class="fact-box">
                        {fact}
                    </div>
                """

            # Use markdown to display the styled fact box
            st.markdown(fact_style, unsafe_allow_html=True)

        # Iterate through the list of facts and display each one
        for fact in facts:
            display_fact(fact)

        # Show latest news on forest fires
    elif options == 'Contact':
        st.write('Contact us at info@example.com.')


# Main app flow
if st.session_state.get('logged_in', False):  # Checks if the 'logged_in' key exists and if it's True
    show_dashboard()  # Show the dashboard if logged in
else:
    if st.session_state.get('view') == "Create Account":
        show_create_account()

    else:
        show_login_page()

        if st.button("Create Account"):
            st.session_state['view'] = "Create Account"
            st.experimental_rerun()
# Function to update the user's password in the database

        # Main app logic
        if 'page' not in st.session_state:
            st.session_state['page'] = 'news'


