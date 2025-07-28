import streamlit as st
import pymongo
import hashlib
import requests
import time
from ai_models import load_and_predict, get_model_instructions, SKIN_DISEASE_CLASSES, BRAIN_DISEASE_CLASSES, EYE_DISEASE_CLASSES, HEART_DISEASE_CLASSES
from face_verification import capture_and_verify, capture_and_signup, FACE_API_URL

MONGODB_URI = "mongodb+srv://projectDB:PEyHwQ2fF7e5saEf@cluster0.43hxo.mongodb.net/"

@st.cache_resource
def init_db():
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        db = client["ta7t-bety"]
        return db["paitents"]
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(collection, name, email, password):
    if collection is None:
        return False, "Failed to connect to database"
    try:
        if collection.find_one({"email": email}):
            return False, "Email already in use"
        user_data = {
            "name": name,
            "email": email,
            "password": hash_password(password),
            "created_at": time.time()
        }
        collection.insert_one(user_data)
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def verify_user(collection, email, password):
    if collection is None:
        return False, None
    try:
        user = collection.find_one({"email": email, "password": hash_password(password)})
        return user is not None, user
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False, None

def display_enhanced_result(result):
    st.markdown("---")
    st.markdown(
        """
        <div style="background:rgba(0,0,0,0.7);border-radius:20px;padding:2rem 1rem 1rem 1rem;box-shadow:0 0 30px #000;">
        <h2 style="background: linear-gradient(90deg, #fff 0%, #b30000 40%, #000 100%); 
                   color: transparent; background-clip: text; -webkit-background-clip: text; 
                   text-shadow: 0 0 20px #b30000, 0 0 40px #fff; 
                   font-weight:bold;letter-spacing:2px;text-align:center;font-size:2.5rem;">
            🔬 AI Medical Result
        </h2>
        </div>
        """, unsafe_allow_html=True
    )
    prediction = result['prediction']
    model_type = result['model_type']
    advice = result['advice']

    st.markdown(
        f"""
        <div style="background:rgba(30,0,0,0.85);border-radius:18px;padding:2rem;margin:1rem 0;box-shadow:0 0 20px #900;">
            <h3 style="color:#fff;font-size:2.2rem;text-align:center;font-weight:bold;">{prediction}</h3>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background:rgba(0,0,0,0.8);border-radius:18px;padding:1.5rem;margin:1rem 0;box-shadow:0 0 10px #222;">
            <h4 style="color:#fff;font-size:1.3rem;font-weight:bold;">💡 Medical Advice</h4>
            <div style="color:#fff;font-size:1.1rem;">{advice}</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background:rgba(30,0,0,0.7);border-radius:12px;padding:1rem;margin:1rem 0;">
            <span style="color:#fff;font-weight:bold;">Disclaimer:</span>
            <span style="color:#fff;">{result['medical_disclaimer']}</span>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="background:rgba(0,0,0,0.8);border-radius:16px;padding:1.5rem 1rem 1rem 1rem;box-shadow:0 0 10px #b30000;">
        <h3 style="color:#fff;font-size:1.5rem;font-weight:bold;margin-bottom:0.7rem;">Emergency Numbers</h3>
        <ul style="color:#fff;font-size:1.1rem;">
        <li><b>Emergency Services:</b> 911 (USA) / 999 (UK) / 112 (Europe)</li>
        <li><b>Poisons Control:</b> 1-800-222-1222 (USA)</li>
        <li><b>Your Doctor's Emergency Number</b></li>
        <li><b>Nearest ER</b></li>
        </ul>
        <p style="color:#fff;font-size:1.1rem;margin-top:1rem;">Call immediately if you experience:</p>
        <ul style="color:#fff;font-size:1.1rem;">
        <li>Severe chest pain or difficulty breathing</li>
        <li>Loss of consciousness or severe confusion</li>
        <li>Heavy bleeding or serious injury</li>
        <li>Signs of stroke (face droop, arm weakness, speech difficulty)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

def show_enhanced_model_instructions(model_type):
    instructions = get_model_instructions(model_type)
    with st.expander("Instructions", expanded=False):
        st.markdown(
            f"""<h2 style="border-bottom: 3px solid #111; padding-bottom: 0.3rem; color: #111; font-weight:bold;letter-spacing:2px;">
            {instructions['title']}
            </h2>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### What to Upload")
            st.markdown(instructions['what_to_send'])
        with col2:
            st.markdown("### What is Detected")
            st.markdown(instructions['what_it_detects'])
        with col3:
            st.markdown("### " + instructions['image_tips'])
        if model_type == "skin-cancer":
            st.markdown("---")
            st.markdown("### Detectable Skin Conditions")
            skin_cols = st.columns(3)
            for i, (class_id, disease_name) in enumerate(SKIN_DISEASE_CLASSES.items()):
                col_idx = i % 3
                with skin_cols[col_idx]:
                    st.write(f"**{class_id}.** {disease_name}")
        elif model_type == "brain-mri":
            st.markdown("---")
            st.markdown("### Detectable Brain Diseases")
            brain_cols = st.columns(2)
            for i, (class_id, disease_name) in enumerate(BRAIN_DISEASE_CLASSES.items()):
                col_idx = i % 2
                with brain_cols[col_idx]:
                    st.write(f"**{class_id}.** {disease_name}")
        elif model_type == "eye-disease":
            st.markdown("---")
            st.markdown("### Detectable Eye Diseases")
            eye_cols = st.columns(2)
            for i, (class_id, disease_name) in enumerate(EYE_DISEASE_CLASSES.items()):
                col_idx = i % 2
                with eye_cols[col_idx]:
                    st.write(f"**{class_id}.** {disease_name}")
        elif model_type == "heart-disease":
            st.markdown("---")
            st.markdown("### Detectable Heart Diseases")
            heart_cols = st.columns(4)
            for i, disease_name in enumerate(HEART_DISEASE_CLASSES):
                col_idx = i % 4
                with heart_cols[col_idx]:
                    st.write(f"**{i+1}.** {disease_name}")

def main():
    st.set_page_config(
        page_title="AI Medical Assistant", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- تعديل حجم صورة الخلفية فقط ليكون أكبر قليلاً ---
    st.markdown("""
    <style>
    body, .stApp {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIJFJpMjKALvJHWYXyByAmY5OPAmmyQRJwsg&s");
        background-size: 80%; /* تم تكبير صورة الخلفية من 30% إلى 45% */
        background-position: center center;
        animation: movebg 30s linear infinite alternate;
    }
    @keyframes movebg {
        0% {background-position: 0 0;}
        100% {background-position: 100px 100px;}
    }
    .login-title, .login-sub, .main-title, .choose-type-title, .face-title, .face-sub {
        background: linear-gradient(90deg, #fff 0%, #b30000 40%, #000 100%);
        color: transparent !important;
        background-clip: text !important;
        -webkit-background-clip: text !important;
        text-shadow: 0 0 20px #b30000, 0 0 40px #fff;
        font-weight: bold;
        letter-spacing: 2px;
        text-align: center;
    }
    .login-title {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    .login-sub {
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        margin-bottom: 1.2rem;
    }
    .choose-type-title {
        font-size: 2rem;
        margin-bottom: 1.2rem;
    }
    .face-title {
        font-size: 2.2rem;
        margin-bottom: 1.2rem;
    }
    .face-sub {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button, .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #fff 0%, #b30000 60%, #000 100%);
        color: #fff !important;
        border: 2px solid #b30000;
        border-radius: 30px !important;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 0 10px #b30000, 0 0 20px #fff;
        transition: 0.2s;
        text-shadow: 0 0 10px #fff, 0 0 20px #b30000;
    }
    .stButton>button:hover, .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(90deg, #fff 0%, #ff0033 60%, #000 100%);
        color: #fff !important;
        border: 2px solid #ff0033;
        box-shadow: 0 0 20px #b30000, 0 0 40px #fff;
    }
    .stTextInput>div>div>input, .stTextInput>div>input {
        background: #111 !important;
        color: #fff !important;
        border: 2px solid #b30000 !important;
        border-radius: 20px !important;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #111 !important;
        color: #fff !important;
        border-radius: 20px 20px 0 0 !important;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background: #b30000 !important;
        color: #fff !important;
    }
    .stAlert {
        border-radius: 20px !important;
        font-size: 1.1rem;
    }
    .stCamera>div {
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
    /* Improved face-overlay: centered, bigger, and lighter for user face visibility */
    .face-overlay {
        position: absolute;
        left: 50%;
        top: 225%;
        width: 600px;
        height: 600px;
        transform: translate(-50%, -50%);
        z-index: 10;
        pointer-events: none;
        display: flex;
        justify-content: center;
        align-items: center;
     }
    .face-overlay img {
        width: 1000px;
        height: 700px;
        border-radius: 1000%;
        box-shadow: 0 0 40px #b30000, 0 0 80px #fff;
        border: 6px solid #b30000;
        object-fit: cover;
        background: rgba(255,255,255,0.25); /* lighter, more transparent */
        opacity: 0.45; /* make overlay lighter so user face is visible */
        mix-blend-mode: lighten;
    }
    @keyframes facepulse {
        0% { box-shadow: 0 0 30px #b30000; transform: scale(1);}
        50% { box-shadow: 0 0 60px #ff0033; transform: scale(1.10);}
        100% { box-shadow: 0 0 30px #b30000; transform: scale(1);}
    }
    .success-check {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        animation: popin 0.7s;
    }
    @keyframes popin {
        0% { transform: scale(0.2); opacity: 0; }
        80% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); }
    }
    .success-check img {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        box-shadow: 0 0 40px #00ff00;
        border: 6px solid #00ff00;
        background: #fff;
        animation: pulsecheck 1.5s infinite;
    }
    @keyframes pulsecheck {
        0% { transform: scale(1);}
        50% { transform: scale(1.18);}
        100% { transform: scale(1);}
    }
    /* Custom chatbot button: use custom icon instead of emoji */
    .chatbot-btn {
        background: linear-gradient(90deg, #b30000 0%, #fff 60%, #ff3300 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 30px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        box-shadow: 0 0 10px #b30000, 0 0 20px #fff, 0 0 10px #ff3300;
        padding: 10px 22px !important;
        margin-top: 10px !important;
        margin-bottom: 8px !important;
        letter-spacing: 1px;
        text-shadow: 0 0 6px #fff, 0 0 10px #b30000;
        transition: all 0.2s;
        animation: chatbotfire 1.2s infinite alternate;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5em;
    }
    .chatbot-btn__icon {
        width: 1.5em;
        height: 1.5em;
        margin-right: 0.5em;
        border-radius: 50%;
        vertical-align: middle;
        display: inline-block;
    }
    .chatbot-btn:hover {
        background: linear-gradient(90deg, #ff3300 0%, #fff 60%, #b30000 100%) !important;
        color: #fff !important;
        box-shadow: 0 0 20px #ff3300, 0 0 40px #fff, 0 0 10px #b30000;
        transform: scale(1.05) rotate(-1deg);
        border: 2px solid #fff !important;
    }
    @keyframes chatbotfire {
        0% { box-shadow: 0 0 10px #b30000, 0 0 20px #fff, 0 0 10px #ff3300;}
        100% { box-shadow: 0 0 20px #ff3300, 0 0 40px #fff, 0 0 20px #b30000;}
    }
    .login-bg {
        background: rgba(0,0,0,0.7);
        border-radius: 30px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin: 2rem auto;
        box-shadow: 0 0 40px #b30000;
        max-width: 500px;
    }
    </style>
    """, unsafe_allow_html=True)

    collection = init_db()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'face_verified' not in st.session_state:
        st.session_state.face_verified = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

    # Login / Signup section
    if not st.session_state.logged_in:
        st.markdown('<div class="login-bg">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Welcome to the AI Medical Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Sign in or create your account to continue</div>', unsafe_allow_html=True)
        if collection is None:
            st.error("❌ Failed to connect to database. Please try again later.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login_btn = st.button("Login Now", key="login_btn", use_container_width=True)
            if login_btn:
                if email and password:
                    success, user_data = verify_user(collection, email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_data = user_data
                        st.markdown(
                            f"""
                            <div class="success-check">
                                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbKYfd2rilINOdT6s0eBxKUWQuEmjwjtumrg&s" alt="Success" />
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.success("Logged in successfully!")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Incorrect email or password")
                else:
                    st.error("Please fill in all fields")
        with tab2:
            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
            name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            signup_btn = st.button("Create New Account", key="signup_btn", use_container_width=True)
            if signup_btn:
                if name and email and password and confirm_password:
                    if password != confirm_password:
                        st.error("❌ Passwords do not match")
                    else:
                        success, message = register_user(collection, name, email, password)
                        if success:
                            st.markdown(
                                f"""
                                <div class="success-check">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbKYfd2rilINOdT6s0eBxKUWQuEmjwjtumrg&s" alt="Success" />
                                </div>
                                """, unsafe_allow_html=True
                            )
                            st.success("Account created successfully!")
                            st.info("👈 Go to the Login tab to sign in")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")
        st.markdown('</div>', unsafe_allow_html=True)

    # Face verification section
    elif not st.session_state.face_verified:
        st.markdown('<div class="login-bg">', unsafe_allow_html=True)
        st.markdown('<div class="face-title">Face Security Check</div>', unsafe_allow_html=True)
        st.markdown('<div class="face-sub">We need to make sure it\'s really you</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Face Verification", "Register Face"])
        with tab1:
            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
            st.info("Use your camera to verify your identity")
            # Centered, bigger, and lighter face overlay for better user face visibility
            st.markdown(
                """
                <div style="width:100%;display:flex;justify-content:center;align-items:center;min-height:480px;">
                    <div class="face-overlay">
                        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMsedlUzk5WULPlKPJOKxLnHgikdfVqpris4P6BFTq1Q6oiAB1vIsX8fnWh-vgqfrlTWI&usqp=CAU" alt="face-shape" />
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            try:
                success, message = capture_and_verify()
                if success:
                    st.session_state.face_verified = True
                    st.markdown(
                        f"""
                        <div class="success-check">
                            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbKYfd2rilINOdT6s0eBxKUWQuEmjwjtumrg&s" alt="Success" />
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.success("Face verified! Welcome!")
                    time.sleep(2)
                    st.rerun()
                elif "No image captured" not in message:
                    # Hide technical messages like "Sending to verify API..." and image size
                    if not ("Sending to verify API" in message or "Image size:" in message):
                        st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Face verification error: {str(e)}")
        with tab2:
            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
            st.info("First time? Add your face to the system")
            st.markdown(
                """
                <div style="width:100%;display:flex;justify-content:center;align-items:center;min-height:480px;">
                    <div class="face-overlay">
                        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMsedlUzk5WULPlKPJOKxLnHgikdfVqpris4P6BFTq1Q6oiAB1vIsX8fnWh-vgqfrlTWI&usqp=CAU" alt="face-shape" />
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            try:
                success, message = capture_and_signup()
                if success:
                    st.markdown(
                        f"""
                        <div class="success-check">
                            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbKYfd2rilINOdT6s0eBxKUWQuEmjwjtumrg&s" alt="Success" />
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.success("Face saved! Go to the 'Face Verification' tab to log in")
                    st.info("Go to the 'Face Verification' tab to log in")
                elif "No image captured" not in message:
                    if not ("Sending to verify API" in message or "Image size:" in message):
                        st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Face registration error: {str(e)}")
        st.markdown("---")
        st.markdown("**System Check:**")
        if st.button("Test Camera System"):
            try:
                response = requests.get(f"{FACE_API_URL}/", timeout=5)
                st.success("✅ Camera system is working!")
            except Exception as e:
                st.error(f"❌ Camera system not working: {str(e)}")
        if st.button("🚪 Log Out"):
            st.session_state.logged_in = False
            st.session_state.face_verified = False
            st.session_state.user_data = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main AI medical section
    else:
        st.markdown(
            """
            <div style="background:rgba(0,0,0,0.7);border-radius:20px;padding:1.5rem 1rem 1rem 1rem;box-shadow:0 0 30px #000;">
            <h2 class="main-title">🏥 AI Medical Assistant</h2>
            </div>
            """, unsafe_allow_html=True
        )
        st.sidebar.title(f"Hello, {st.session_state.user_data['name']}")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Links")
        st.sidebar.markdown(
            """
            - [CDC Health Topics](https://www.cdc.gov/health-topics.html)
            - [WHO Health Topics](https://www.who.int/health-topics)
            - [Find a Doctor](https://www.healthgrades.com/)
            - [First Aid Guide](https://www.redcross.org/get-help/first-aid/first-aid-steps.html)
            """,
            unsafe_allow_html=True
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Emergency Numbers")
        st.sidebar.markdown(
            """
            - **Emergency Services:** 911 (USA) / 999 (UK) / 112 (Europe)
            - **Poisons Control:** 1-800-222-1222 (USA)
            - **Your Doctor's Emergency Number**
            - **Nearest ER**
            """,
            unsafe_allow_html=True
        )
        st.sidebar.markdown("---")
        if st.sidebar.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.face_verified = False
            st.session_state.user_data = None
            st.rerun()
        medical_categories = {
            "🫁 Chest X-ray Analysis": "chest-xray",
            "🦴 Bone Fracture Detection": "bone-fracture", 
            "🔬 Skin Disease Classification": "skin-cancer",
            "👁️ Eye Disease Detection": "eye-disease",
            "🧠 Brain MRI Analysis": "brain-mri",
            "❤️ Heart Disease Detection": "heart-disease"
        }
        st.markdown('<div class="choose-type-title">Choose Medical Analysis Type</div>', unsafe_allow_html=True)
        selected_category = st.selectbox(
            "What type of medical test do you want to analyze?", 
            list(medical_categories.keys()),
            help="Upload your medical test for AI analysis."
        )
        if selected_category:
            model_type = medical_categories[selected_category]
            st.markdown(
                f"""
                <div style="background:linear-gradient(90deg,#fff 0%,#b30000 60%,#000 100%);color:transparent;
                background-clip:text;-webkit-background-clip:text;text-shadow:0 0 20px #b30000,0 0 40px #fff;
                font-weight:bold;padding:10px 20px;border-radius:20px;display:inline-block;margin:10px 0;font-size:1.5rem;">
                {selected_category}
                </div>
                """, unsafe_allow_html=True
            )
            show_enhanced_model_instructions(model_type)
            st.markdown("---")
            st.subheader(selected_category)
            uploader_help = "Upload a clear medical image for AI analysis."
            if model_type == "chest-xray":
                uploader_help += " For pneumonia analysis, ensure high-quality X-ray image."
            elif model_type == "skin-cancer":
                uploader_help += " For skin analysis, use dermoscopy images if possible."
            elif model_type == "heart-disease":
                uploader_help += " Note: Heart model requires ECG signal data, not images."
            if model_type == "heart-disease":
                uploaded_file = st.file_uploader(
                    "Upload your ECG data file", 
                    type=['hea', 'mat'],
                    help="Upload ECG files (.hea for info, .mat for signal)."
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload your medical image", 
                    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                    help=uploader_help
                )
            if uploaded_file:
                if model_type == "heart-disease":
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(
                            f"""
                            <div style="border:2px dashed #b30000;padding:20px;border-radius:10px;text-align:center;background:rgba(0,0,0,0.7);color:#fff;">
                                <h3>ECG File Uploaded</h3>
                                <p><strong>File:</strong> {uploaded_file.name}</p>
                                <p><strong>Type:</strong> {uploaded_file.type if uploaded_file.type else 'ECG Data'}</p>
                                <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                                <p><em>Ready for analysis</em></p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown("### File Details")
                        st.write(f"**File Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                        st.write(f"**Format:** {uploaded_file.name.split('.')[-1].upper()}")
                else:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)
                    with col2:
                        st.markdown("### Image Info")
                        st.write(f"**File Name:** {uploaded_file.name}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                        st.write(f"**Type:** {uploaded_file.type}")
                button_text = f"🔬 Analyze"
                if st.button(button_text, use_container_width=True, type="primary"):
                    with st.spinner("🤖 AI is analyzing... Please wait"):
                        try:
                            result = load_and_predict(uploaded_file, model_type)
                            display_enhanced_result(result)
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.info("Try uploading another file or contact support")
        st.markdown("---")
        st.subheader("Need More Help?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 Questions about AI?")
            st.markdown("""
            **Frequently Asked Questions:**
            - How to get the best results?
            - What to do with the results?
            """)
        with col2:
            st.markdown("""
            <a href="https://share.chatling.ai/s/8T75tziUCDxg26K" target="_blank">
                <button class="chatbot-btn">
                    🤖 Chatbot Assistant
                </button>
            </a>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            """
            <div style="background:rgba(30,0,0,0.7);border-radius:12px;padding:1rem;margin:1rem 0;">
                <span style="color:#fff;font-weight:bold;">Disclaimer:</span>
                <span style="color:#fff;">This system is for triage and education only and does not replace professional medical diagnosis. Always consult qualified physicians before making any medical decisions.</span>
            </div>
            """, unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
