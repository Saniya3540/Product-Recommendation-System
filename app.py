import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
dataset = pickle.load(open("products.pkl", "rb"))
vectors = pickle.load(open("vectors.pkl", "rb"))

# Recommendation function
def recommend(product_name):
    try:
        index = dataset[dataset['Product'] == product_name].index[0]
    except:
        return []

    distances = cosine_similarity(vectors[index], vectors).flatten()

    product_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recs = []
    seen = set()

    for i in product_list:
        prod = dataset.iloc[i[0]]['Product']

        # avoid duplicates + same product
        if prod != product_name and prod not in seen:
            recs.append(prod)
            seen.add(prod)

        if len(recs) == 5:
            break

    return recs


# UI
st.title("🛒 Product Recommendation System")

query = st.text_input("🔍 Enter product or brand")

if query:
    results = dataset[
        dataset['Product'].str.lower().str.contains(query.lower(), na=False) |
        dataset['Brand'].str.lower().str.contains(query.lower(), na=False)
    ]

    if results.empty:
        st.error("❌ No products found")
    else:
        product_list = results['Product'].unique()

        selected_product = st.selectbox("📦 Select product", product_list)

        if selected_product:
            st.subheader("✅ Selected Product")
            st.write(selected_product)

            # Reviews
            st.subheader("📝 Reviews")
            reviews = dataset[dataset['Product'] == selected_product]['Review'].head(3)

            if len(reviews) == 0:
                st.write("No reviews available")
            else:
                for r in reviews:
                    st.write("•", r)

            # Recommendations
            st.subheader("💡 Recommended Products")
            recs = recommend(selected_product)

            if len(recs) == 0:
                st.write("No recommendations found")
            else:
                for r in recs:
                    st.write("👉", r)