import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from PIL import Image
import random

# ======================
# APP CONFIGURATION
# ======================
st.set_page_config(
    page_title="Gourmet Guide Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS
# ======================
def inject_custom_css():
    st.markdown("""
    <style>
        :root {
            --primary: #FF4B4B;
            --secondary: #FF9A3C;
            --accent: #20C0E0;
            --dark: #1A1A2E;
            --light: #F5F5F5;
            --success: #4CAF50;
        }
        
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #f0f2f5 100%);
        }
        
        .stSidebar {
            background: var(--dark) !important;
            color: white !important;
            padding: 1.5rem !important;
        }
        
        .sidebar-title {
            color: var(--primary) !important;
            font-size: 1.8rem !important;
            margin-bottom: 1.5rem !important;
        }
        
        .card {
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background: white;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            border-left: 4px solid var(--primary);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
        }
        
        .price-tag {
            background: var(--primary);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
        }
        
        .rating-badge {
            background: var(--secondary);
            color: white;
            border-radius: 20px;
            padding: 0.3rem 0.8rem;
            font-weight: bold;
            display: inline-block;
            margin-right: 0.5rem;
        }
        
        .cuisine-tag {
            background: #E0E0E0;
            border-radius: 20px;
            padding: 0.3rem 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            display: inline-block;
            font-size: 0.8rem;
        }
        
        .primary-button {
            background: var(--primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.7rem 1.5rem !important;
            font-weight: bold !important;
            transition: all 0.3s !important;
        }
        
        .primary-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3) !important;
        }
        
        .section-title {
            color: var(--dark);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        .hero {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 3rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ======================
# DATA LOADING & PROCESSING
# ======================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("Dataset.csv")
    except FileNotFoundError:
        st.error("🚨 Dataset file not found. Please ensure 'Dataset.csv' is in the same directory.")
        st.stop()
    
    # Data cleaning
    df.dropna(subset=['Cuisines', 'Price range', 'Aggregate rating'], inplace=True)
    df['Cuisines'] = df['Cuisines'].str.lower().str.strip()
    df['City'] = df['City'].fillna('Unknown')
    
    # Pre-process for faster filtering
    df['Cuisine_List'] = df['Cuisines'].str.split(',')
    df['Cuisine_List'] = df['Cuisine_List'].apply(lambda x: [c.strip() for c in x])
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def compute_similarity(_df):
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf_matrix = tfidf.fit_transform(_df['Cuisines'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Load data with progress
with st.spinner('🧑‍🍳 Loading restaurant database...'):
    df = load_data()
    cos_sim = compute_similarity(df)

# ======================
# UI COMPONENTS
# ======================
def hero_section():
    st.markdown("""
    <div class="hero">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Gourmet Guide Pro</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Discover your perfect dining experience with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)

def filters_sidebar():
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">🍽️ Filters</h1>', unsafe_allow_html=True)
        
        # Location Filter
        selected_city = st.selectbox(
            "📍 City", 
            sorted(df['City'].unique()),
            index=0,
            help="Select your preferred city"
        )
        
        # Cuisine Filter
        all_cuisines = sorted({c.strip().lower() for row in df['Cuisines'] for c in row.split(',')})
        selected_cuisines = st.multiselect(
            "🍜 Cuisines", 
            all_cuisines,
            default=[],
            help="Select up to 5 cuisines",
            max_selections=5
        )
        
        # Dietary Preference
        diet_pref = st.selectbox(
            "🌱 Dietary Needs", 
            ["No Preference", "Vegetarian", "Vegan", "Gluten-Free", "Halal", "Kosher"],
            index=0
        )
        
        # Price Range
        price = st.slider(
            "💰 Price Range", 
            1, 4, (1, 4),
            help="1 = Budget, 4 = Fine Dining"
        )
        
        # Rating Filter
        min_rating = st.slider(
            "⭐ Minimum Rating", 
            0.0, 5.0, 3.5, 0.1,
            format="%.1f stars"
        )
        
        # Quick Filters
        st.markdown("### ⚡ Quick Presets")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Top Rated", help="Show only 4.5+ rated restaurants"):
                st.session_state.min_rating = 4.5
                st.experimental_rerun()
        with col2:
            if st.button("Budget Eats", help="Show affordable options (Price 1-2)"):
                st.session_state.price = (1, 2)
                st.experimental_rerun()
        
        return selected_city, selected_cuisines, diet_pref, price, min_rating

def restaurant_card(name, cuisines, rating, price, delay=0):
    price_icons = "💲" * price
    cuisine_tags = "".join([f'<span class="cuisine-tag">{c.strip()}</span>' for c in cuisines.split(',')])
    
    return f"""
    <div class="card animate" style="animation-delay: {delay}s">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 style="margin: 0;">{name}</h3>
            <span class="price-tag">{price_icons}</span>
        </div>
        <div style="margin-bottom: 0.8rem;">
            <span class="rating-badge">{rating} ⭐</span>
        </div>
        <div style="margin-bottom: 0.5rem;">
            {cuisine_tags}
        </div>
    </div>
    """

# ======================
# RECOMMENDATION ENGINE
# ======================
@st.cache_data(show_spinner=False)
def recommend(_df, _cos_sim, city, cuisines, diet, price_range, rating, top_n=12):
    # Start with city filter
    mask = (_df['City'] == city)
    
    # Apply cuisine filter if any selected
    if cuisines:
        cuisine_mask = _df['Cuisine_List'].apply(lambda x: any(c in x for c in cuisines))
        mask &= cuisine_mask
    
    # Apply dietary filters
    diet_filters = {
        "Vegetarian": ['vegetarian', 'veg'],
        "Vegan": ['vegan'],
        "Gluten-Free": ['gluten-free', 'gluten free'],
        "Halal": ['halal'],
        "Kosher": ['kosher']
    }
    
    if diet in diet_filters:
        mask &= _df['Cuisines'].str.contains('|'.join(diet_filters[diet]))
    
    # Apply price and rating filters
    mask &= (_df['Price range'] >= price_range[0]) & (_df['Price range'] <= price_range[1])
    mask &= (_df['Aggregate rating'] >= rating)
    
    filtered_df = _df[mask].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Get similarity scores for filtered restaurants
    indices = filtered_df.index.values
    similarity_scores = []
    
    for idx in indices:
        # Get top 3 most similar restaurants (excluding self)
        sim_scores = list(enumerate(_cos_sim[idx]))
        sim_scores = [(i, score) for i, score in sim_scores if i != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        if sim_scores:
            similarity_scores.append((idx, np.mean([s[1] for s in sim_scores[:3]])))
    
    # Sort by similarity score then by rating
    filtered_df['Similarity'] = filtered_df.index.map(dict(similarity_scores))
    filtered_df = filtered_df.sort_values(['Similarity', 'Aggregate rating'], ascending=[False, False])
    
    return filtered_df.head(top_n)[['Restaurant Name', 'City', 'Cuisines', 
                                 'Aggregate rating', 'Price range']]

# ======================
# MAIN APP
# ======================
def main():
    hero_section()
    city, cuisines, diet, price, min_rating = filters_sidebar()
    
    # Main recommendation button
    if st.button("✨ Find My Perfect Restaurants", key="find_restaurants", use_container_width=True):
        with st.spinner('🔍 Searching for culinary gems...'):
            # Simulate processing for better UX
            time.sleep(0.8)
            
            results = recommend(df, cos_sim, city, cuisines, diet, price, min_rating)
            
            if not results.empty:
                st.balloons()
                st.markdown(f"""
                <h2 class="section-title" style="margin-top: 1.5rem;">
                    🎉 We found {len(results)} perfect matches in {city}
                </h2>
                """, unsafe_allow_html=True)
                
                # Display results in a responsive grid
                cols = st.columns(2)
                for idx, (_, row) in enumerate(results.iterrows()):
                    with cols[idx % 2]:
                        st.markdown(restaurant_card(
                            row['Restaurant Name'],
                            row['Cuisines'],
                            row['Aggregate rating'],
                            row['Price range'],
                            delay=idx*0.1
                        ), unsafe_allow_html=True)
            else:
                st.warning("""
                ## 🕵️‍♂️ No restaurants matched your criteria
                Try adjusting your filters or expanding your search parameters
                """)
                
                # Show popular alternatives
                popular = df[
                    (df['City'] == city) & 
                    (df['Aggregate rating'] >= 4.0)
                ].sort_values('Aggregate rating', ascending=False).head(3)
                
                if not popular.empty:
                    st.info("### 💡 Popular alternatives in your area:")
                    for _, row in popular.iterrows():
                        st.markdown(f"""
                        - **{row['Restaurant Name']}**  
                          {row['Cuisines'].split(',')[0]} • {row['Aggregate rating']}⭐ • {'💲' * int(row['Price range'])}
                        """)
    
    # Trending section
    st.markdown("""
    <h2 class="section-title">
        🔥 Trending This Week
    </h2>
    """, unsafe_allow_html=True)
    
    trending = df[df['Aggregate rating'] >= 4.3].sample(min(4, len(df[df['Aggregate rating'] >= 4.3])))
    cols = st.columns(4)
    for idx, (_, row) in enumerate(trending.iterrows()):
        with cols[idx % 4]:
            st.markdown(restaurant_card(
                row['Restaurant Name'],
                row['Cuisines'],
                row['Aggregate rating'],
                row['Price range'],
                delay=0
            ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()