import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="🛍️ Product Similarity Search", layout="centered")
st.title("🛍️ Amazon Product Recommender")
st.markdown("Enter a product description or name to find similar items from the product catalog.")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "product_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

query = st.text_input("🔍 What product are you looking for?")

if query:
    st.subheader("🧠 Top Similar Products")
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        st.markdown(f"### {i}.")
        st.write(doc.page_content)
