"""
🍳 Smart Chef — Phase 7 Streamlit App 
· Content-based recipe recommender
· Cuisine classifier
· Meal-time classifier
· Course classifier
· Cooking-time predictor
"""

import os
import gzip
#import subprocess
import streamlit as st  # Assuming you already import streamlit at the top

# --- NEW AUTOMATIC DATA GENERATION ---
#  We check if the model exists. If it doesn't, we run the scripts.
#  if not os.path.exists('models/svd_model.pkl'):
#  with st.spinner("First time setup: Generating data and training models. This will take a few minutes..."):
#  try:
#  Run the scripts in order
#  subprocess.run(["python", "data_collection.py"], check=True)
#  subprocess.run(["python", "data_cleaning.py"], check=True)
#  subprocess.run(["python", "feature_engineering.py"], check=True)
#  Add any other script that creates your models here (e.g., recommendation.py)
#  subprocess.run(["python", "recommendation.py"], check=True)     
#  st.success("Setup complete! Loading app...")
#  except subprocess.CalledProcessError as e:
#  st.error(f"Error during auto-generation: {e}")
#  st.stop()

# Force clear proxy variables to prevent httpx parsing errors
for _var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
             "http_proxy", "https_proxy", "all_proxy", "no_proxy"]:
    os.environ.pop(_var, None)

from google import genai
from dotenv import load_dotenv

# Load the API key
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Chef 🍳",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');

/* ============ FORCE DARK EVERYWHERE ============ */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], section.main {
    background-color: #0f0f1a !important;
    color: #f5f5f5 !important;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 15%, #2d1b3d 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, #3d1f1f 0%, transparent 45%),
        linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%) !important;
    color: #f5f5f5;
}

/* ============ HERO ============ */
.hero-title {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7e5f, #feb47b, #ffd700, #ff7e5f);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    animation: shine 4s ease-in-out infinite;
}
@keyframes shine {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.hero-sub {
    text-align: center;
    color: #b8b8d1;
    font-size: 1.15rem;
    margin-top: -12px;
    margin-bottom: 35px;
    letter-spacing: 0.5px;
}

/* ============ RECIPE CARDS ============ */
.recipe-card {
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 14px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.recipe-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff7e5f, #feb47b, #ffd700);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.recipe-card:hover {
    transform: translateY(-6px);
    border-color: rgba(255, 126, 95, 0.5);
    box-shadow: 0 15px 40px rgba(255, 126, 95, 0.18);
    background: rgba(255, 255, 255, 0.07);
}
.recipe-card:hover::before { opacity: 1; }

.recipe-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffd6a5;
    margin-bottom: 8px;
    line-height: 1.35;
}
.recipe-meta {
    color: #b8b8d1;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ============ BADGES ============ */
.badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(255,126,95,0.25), rgba(254,180,123,0.25));
    border: 1px solid rgba(255,126,95,0.4);
    color: #ffd6a5;
    font-weight: 500;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.76rem;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ============ RESULT CARDS ============ */
.result-card {
    background: linear-gradient(135deg, rgba(255,126,95,0.12), rgba(255,215,0,0.08));
    border: 1px solid rgba(255,126,95,0.3);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    backdrop-filter: blur(14px);
    animation: fadeIn 0.6s ease-out;
    margin-bottom: 12px;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    color: #b8b8d1;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7e5f, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

/* ============ BUTTONS ============ */
.stButton > button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: #1a1a2e;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 1rem;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 14px rgba(255, 126, 95, 0.25);
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 10px 28px rgba(255, 126, 95, 0.45);
    color: #1a1a2e;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(255, 255, 255, 0.03);
    padding: 8px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 11px;
    padding: 11px 20px;
    color: #b8b8d1;
    font-weight: 500;
    font-size: 0.95rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
}

/* ============ METRICS ============ */
[data-testid="stMetricValue"] { color: #ffd6a5; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #b8b8d1; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px 18px;
}

/* ============ TEXT AREAS ============ */
textarea,
div[data-testid="stTextArea"] textarea,
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    caret-color: #ff7e5f !important;
    font-family: 'Poppins', sans-serif !important;
}
textarea::placeholder {
    color: #8a8aa3 !important;
    -webkit-text-fill-color: #8a8aa3 !important;
    opacity: 1 !important;
}
textarea:focus {
    border-color: #ff7e5f !important;
    box-shadow: 0 0 0 3px rgba(255, 126, 95, 0.18) !important;
}

/* ============ SELECT INPUT (closed state) ============ */
input[role="combobox"],
input[aria-autocomplete="list"],
.stSelectbox input {
    background-color: rgba(255, 255, 255, 0.06) !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    font-family: 'Poppins', sans-serif !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="combobox"] > div,
div[data-baseweb="input"] {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
}

/* ================================================================== */
/*  NUCLEAR FIX: kill every white popup Streamlit renders             */
/*  Targets the portal at the end of <body> where the dropdown lives  */
/* ================================================================== */

/* The popup container (BaseWeb portal) */
div[data-baseweb="popover"] {
    background-color: #1a1a2e !important;
    border: 1px solid rgba(255, 126, 95, 0.35) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
    color: #f5f5f5 !important;
}

/* Absolutely every descendant of the popover — universal * selector */
div[data-baseweb="popover"] * {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    background-image: none !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    border-color: rgba(255, 126, 95, 0.15) !important;
}

/* Any listbox / option anywhere on the page */
[role="listbox"],
[role="listbox"] *,
ul[role="listbox"],
ul[role="listbox"] *,
li[role="option"],
li[role="option"] *,
[role="option"],
[role="option"] * {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    background-image: none !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Hover & selected states — must come AFTER the blanket rules */
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
ul[role="listbox"] li:hover,
li[role="option"]:hover,
[role="option"]:hover,
[aria-selected="true"] {
    background-color: rgba(255, 126, 95, 0.3) !important;
    background: rgba(255, 126, 95, 0.3) !important;
    color: #ffd6a5 !important;
    -webkit-text-fill-color: #ffd6a5 !important;
}

/* ============ HIDE STREAMLIT CHROME ============ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

h3 { color: #ffd6a5; font-weight: 700; margin-bottom: 18px; }

label, .stTextArea label, .stSelectbox label {
    color: #b8b8d1 !important;
    font-weight: 500 !important;
}
    /* ============ DROPDOWN POPUP — CONFIRMED WORKING ============ */
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
[role="listbox"],
[role="listbox"] *,
[role="option"],
[role="option"] * {
    background: #1a1a2e !important;
    background-color: #1a1a2e !important;
    background-image: none !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Hover state (must come after the blanket rule) */
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
[role="listbox"] li:hover,
[role="option"]:hover {
    background: rgba(255, 126, 95, 0.3) !important;
    background-color: rgba(255, 126, 95, 0.3) !important;
    color: #ffd6a5 !important;
    -webkit-text-fill-color: #ffd6a5 !important;
}

/* Popup border & shadow */
div[data-baseweb="popover"] {
    border: 1px solid rgba(255, 126, 95, 0.35) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
}
    /* ============ OPAQUE DARK FOR ALL INPUTS (fixes white bars) ============ */

/* Text inputs / combobox / textarea — solid dark bg, not translucent */
input[type="text"],
input[role="combobox"],
input[aria-autocomplete="list"],
textarea,
.stTextInput input,
.stTextArea textarea,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="base-input"] > div,
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] > div,
div[data-baseweb="combobox"],
div[data-baseweb="combobox"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    background-image: none !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
    border-color: rgba(255, 126, 95, 0.25) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* The wrapper div around the textarea / input */
div[data-testid="stTextArea"] > div,
div[data-testid="stTextArea"] > div > div,
.stTextArea > div,
.stTextArea > div > div,
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] > div > div {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    border-radius: 12px !important;
}

/* Focus state — keep the orange glow but on solid dark bg */
input[type="text"]:focus,
input[role="combobox"]:focus,
textarea:focus,
div[data-testid="stTextArea"] textarea:focus {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    border-color: #ff7e5f !important;
    box-shadow: 0 0 0 3px rgba(255, 126, 95, 0.22) !important;
    color: #f5f5f5 !important;
    -webkit-text-fill-color: #f5f5f5 !important;
}

/* Placeholder text — visible on dark */
input::placeholder,
textarea::placeholder,
input[role="combobox"]::placeholder {
    color: #8a8aa3 !important;
    -webkit-text-fill-color: #8a8aa3 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# LOAD ARTIFACTS
@st.cache_resource(show_spinner="🔥 Loading recipes & models...")
def load_all():
    df = pd.read_csv('data/dummy_recipes.csv')
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))

    with gzip.open('models/cuisine_model.pkl', 'rb') as f:
        cuisine_clf, cuisine_vec = pickle.load(f)
    with gzip.open('models/meal_time_model.pkl', 'rb') as f:
        meal_clf, meal_vec = pickle.load(f)
    with gzip.open('models/course_model.pkl', 'rb') as f:
        course_clf, course_vec = pickle.load(f)

    with gzip.open('models/regression_model.pkl', 'rb') as f:
        reg, reg_vec = pickle.load(f)

    # rec_vec = TfidfVectorizer(max_features=3000, min_df=5, stop_words='english')
    # rec_matrix = rec_vec.fit_transform(df['ingredients_str'])

    import scipy.sparse
    # Creae a dummy matrix of zeros so the app does not crash when it loads.
    rec_matrix = scipy.sparse.csr_matrix((len(df), 3000))

    return (
        df,
        cuisine_clf, cuisine_vec,
        meal_clf, meal_vec,
        course_clf, course_vec,
        reg, reg_vec,
        rec_vec, rec_matrix,
    )


(
    df,
    cuisine_clf, cuisine_vec,
    meal_clf, meal_vec,
    course_clf, course_vec,
    reg, reg_vec,
    rec_vec, rec_matrix,
) = load_all()

# HERO
st.markdown('<h1 class="hero-title">🍳 Smart Chef</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Your AI-powered recipe assistant · 228,000+ recipes at your fingertips</p>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("📚 Recipes", f"{len(df):,}")
c2.metric("🥕 Max Ingredients", f"{df['n_ingredients'].max():.0f}")
c3.metric("📋 Max Steps", f"{df['n_steps'].max():.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs([
    "🔍 Similar Recipes",
    "🏷️ Recipe Classifier",
    "⏱️ Time Predictor",
])

# TAB 1 — Recommender
with tab1:
    st.markdown("### Find recipes similar to one you already love")

    # Build a curated list of clean recipe names
    recipe_pool = df.dropna(subset=['name']).copy()
    recipe_pool = recipe_pool[~recipe_pool['name'].str.strip().str[0].str.isdigit()]
    recipe_pool = recipe_pool[recipe_pool['name'].str.len().between(3, 40)]
    recipe_pool = recipe_pool[recipe_pool['n_ingredients'].between(4, 15)]
    recipe_pool = recipe_pool.drop_duplicates(subset='name')
    recipe_pool = recipe_pool.sort_values('n_ingredients', ascending=False)

    # --- NEW PAKISTANI DISHES FILTER ---
    local_only = st.checkbox("Show only Local Pakistani & Desi Dishes")

    if local_only:
        # Filter the dataframe for Pakistani/South Asian keywords
        pak_keywords = 'biryani|nihari|karahi|haleem|tikka|kebab|samosa|paratha|daal|roti|pulao|chutney|masala|korma|chaat|halwa|kheer'
        recipe_pool = recipe_pool[recipe_pool['name'].str.contains(pak_keywords, case=False, na=False)]

    popular_names = recipe_pool['name'].head(500).tolist()

    # Prevent crash if no recipes match
    if not popular_names:
        st.warning("No recipes found for this filter. Try unchecking the box.")
        st.stop()

    col1, col2 = st.columns([4, 1])
    with col1:
        recipe_name = st.selectbox(
            "Pick a recipe:",
            options=popular_names,
            key="selected_recipe"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Recommend"):
            st.session_state['show_results'] = True

    if st.session_state.get('show_results', False):
        with st.spinner("Finding matches..."):
            idx = df[df['name'] == recipe_name].index[0]
            sims = cosine_similarity(rec_matrix[idx], rec_matrix).flatten()
            top_idx = sims.argsort()[::-1][1:11]

        st.markdown(f"#### Top 10 similar to *{recipe_name}*")
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)

        for i, tidx in enumerate(top_idx, 1):
            row = df.iloc[tidx]
            card = f"""
            <div class="recipe-card">
                <div class="recipe-title">{i}. {row['name']}</div>
                <div class="recipe-meta">
                    <span class="badge">⏱ {row['minutes']} min</span>
                    <span class="badge">🥕 {row['n_ingredients']} ingr</span>
                    <span class="badge">📋 {row['n_steps']} steps</span>
                </div>
            </div>
            """
            with (left if i % 2 == 1 else right):
                st.markdown(card, unsafe_allow_html=True)

    # --- NEW GENERATIVE AI SECTION ---
    st.markdown("---")
    st.markdown("### 🤖 AI Chef Assistant")
    st.write(f"Need help cooking **{recipe_name}**? Ask our AI Chef for tips or a quick substitution!")

    # Let the user ask a custom question
    ai_prompt = st.text_input("Ask a question about this recipe:")

    if st.button("✨ Generate Chef Tip"):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("API Key missing! Please check your .env file.")
        elif not ai_prompt:
            st.warning("Please type a question first!")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    prompt = f"I am cooking '{recipe_name}'. {ai_prompt}. Give me a short, helpful, and friendly chef's tip."
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt
                    )
                    st.info(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

# TAB 2 — Multi-Target Classifier
with tab2:
    st.markdown("### Classify your recipe on three axes")

    # 1. Get the recipe selected in Tab 1
    if "selected_recipe" in st.session_state:
        chosen_recipe = st.session_state["selected_recipe"]
        st.markdown(f"#### 🍲 Recipe: *{chosen_recipe}*")

        # 2. Find that exact recipe in the dataset
        recipe_row = df[df['name'] == chosen_recipe].iloc[0]

        # 3. Extract the ingredients silently in the background
        try:
            if isinstance(recipe_row['ingredients'], list):
                actual_ingredients = " ".join([str(ing).strip().replace(" ", "_") for ing in recipe_row['ingredients']])
            else:
                actual_ingredients = str(recipe_row['ingredients']).strip().replace(" ", "_")
        except:
            actual_ingredients = "tomato mozzarella basil olive_oil"
    else:
        st.markdown("#### 🍲 Recipe: *Default*")
        actual_ingredients = "tomato mozzarella basil olive_oil"

    # Note: The text_area is completely removed here!

    if st.button("🏷️ Classify Recipe"):
        with st.spinner("Analyzing..."):
            text = actual_ingredients.lower()

            c_pred = cuisine_clf.predict(cuisine_vec.transform([text]))[0]
            c_probs = cuisine_clf.predict_proba(cuisine_vec.transform([text]))[0]

            m_pred = meal_clf.predict(meal_vec.transform([text]))[0]
            m_probs = meal_clf.predict_proba(meal_vec.transform([text]))[0]

            k_pred = course_clf.predict(course_vec.transform([text]))[0]
            k_probs = course_clf.predict_proba(course_vec.transform([text]))[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🍝 Cuisine</div>
                <div class="result-value">{c_pred.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">⏰ Meal-Time</div>
                <div class="result-value">{m_pred.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">🍰 Course</div>
                <div class="result-value">{k_pred.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["Cuisine confidence", "Meal-time confidence", "Course confidence"])
        with t1:
            st.bar_chart(
                pd.DataFrame({'Probability': c_probs}, index=cuisine_clf.classes_),
                color="#ff7e5f", height=300,
            )
        with t2:
            st.bar_chart(
                pd.DataFrame({'Probability': m_probs}, index=meal_clf.classes_),
                color="#ff7e5f", height=300,
            )
        with t3:
            st.bar_chart(
                pd.DataFrame({'Probability': k_probs}, index=course_clf.classes_),
                color="#ff7e5f", height=300,
            )

# TAB 3 — Cooking Time Predictor
with tab3:
    st.markdown("### How long will this take to cook?")

    # 1. Get the recipe selected in Tab 1
    if "selected_recipe" in st.session_state:
        chosen_recipe = st.session_state["selected_recipe"]

        st.markdown(f"#### 🍲 Recipe: *{chosen_recipe}*")

        # 2. Find that exact recipe in the dataset
        recipe_row = df[df['name'] == chosen_recipe].iloc[0]

        # 3. Safely extract and format ingredients
        try:
            if isinstance(recipe_row['ingredients'], list):
                formatted_ingredients = [str(ing).strip().replace(" ", "_") for ing in recipe_row['ingredients']]
                actual_ingredients = ", ".join(formatted_ingredients)
            else:
                actual_ingredients = str(recipe_row['ingredients']).strip().replace(" ", "_")
        except:
            actual_ingredients = "chicken, garlic, onion, tomato, olive_oil"
    else:
        st.markdown("#### 🍲 Recipe: *Default*")
        actual_ingredients = "chicken, garlic, onion, tomato, olive_oil"

    # 4. Calculate the ingredient count immediately
    clean_input_for_counting = actual_ingredients.replace(",", " ")
    n_ing = len(clean_input_for_counting.split())

    # 5. Show the count right above the text area so you see it instantly
    st.info(f"📊 This recipe contains **{n_ing}** ingredients.")

    # 6. The text area
    ing_input = st.text_area(
        "Ingredients:",
        value=actual_ingredients,
        height=110
    )

    # 7. Updated Button Text
    if st.button("⏱️ Predict Time & Confirm Ingredients"):
        with st.spinner("Estimating..."):
            # Recalculate in case you manually edited the text box
            clean_input_for_counting = ing_input.replace(",", " ")
            n_ing = len(clean_input_for_counting.split())

            clean_input_for_model = ing_input.replace(",", " ").lower()

            ing_vec = reg_vec.transform([clean_input_for_model])
            numeric = np.array([[n_ing, sum(len(i) for i in clean_input_for_model.split())]])
            X = hstack([ing_vec, numeric]).tocsr()
            pred = reg.predict(X)[0]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Cooking Time</div>
            <div class="result-value" style="font-size:2.8rem;">{pred:.0f} min</div>
            <div class="recipe-meta" style="margin-top:10px;">
                Based on <b>{n_ing}</b> ingredient{'s' if n_ing != 1 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#6b6b85;font-size:0.85rem;">'
    'Built with ❤️ using Streamlit ·'
    '</p>',
    unsafe_allow_html=True,
)