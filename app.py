import streamlit as st
from bert_model import BERTSimilarity
from wordcloud_gen import generate_wordcloud

st.set_page_config(page_title="BERT Similarity Search", page_icon="📄")
st.title("📜 Word Similarity Search with BERT Embeddings", )

st.badge("Using bert-base-uncased", icon=":material/check_box:", color="green")

input_word = st.text_input("Enter a word:")
top_n = st.slider("Number of similar words", 1, 50, 10)

@st.cache_resource
def load_model():
    return BERTSimilarity()

model = load_model()

if input_word:
    similar_words = model.get_similar_words(input_word, top_n)
    st.subheader("Similar Words:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        for i in range(0, len(similar_words), 4):
            st.write(f"- **{similar_words[i][0]}** ({similar_words[i][1]:.3f})")
    with col2:
        for i in range(1, len(similar_words), 4):
            st.write(f"- **{similar_words[i][0]}** ({similar_words[i][1]:.3f})")        
    with col3:
        for i in range(2, len(similar_words), 4):
            st.write(f"- **{similar_words[i][0]}** ({similar_words[i][1]:.3f})")
    with col4:
        for i in range(3, len(similar_words), 4):
            st.write(f"- **{similar_words[i][0]}** ({similar_words[i][1]:.3f})")
    
    st.divider()

    st.subheader("Word Cloud:")
    generate_wordcloud(similar_words)
