from PIL.ImageColor import colormap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

def generate_wordcloud(word_scores):
    word_freq = {word: score for word, score in word_scores}
    wc = WordCloud(width=1600, height=800, background_color="black", colormap='Set3').generate_from_frequencies(word_freq)
    st.image(wc.to_array(), use_container_width=True)
