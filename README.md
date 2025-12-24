# 📜 Word Similarity Search with BERT Embeddings

A simple and interactive `Streamlit` app that finds semantically similar words using BERT embeddings and visualizes them in a word cloud. Powered by the `bert-base-uncased` model from Hugging Face Transformers.

## 🚀 Features

- 🔍 Search for semantically similar words using contextual BERT embeddings
- 📊 Visualize results as a multi-column list with similarity scores
- ☁️ Generate an interactive word cloud based on the top similar words
- 🕸️ [LATEST] Visualize semantic similarity of words through an interactive network of graph
- ✅ Efficient model loading using `@st.cache_resource`

## 🧠 Model

This app uses a pre-trained `bert-base-uncased` model from Hugging Face to compute word embeddings and cosine similarity for word comparison.

## 🖼️ Demo

Enter a word in the input box, choose how many similar words to retrieve (1–50), and view:

- A list of similar words with similarity scores
- A word cloud representing the output distribution

[LATEST]
- Accepts a word, top_n, and graph similarity threshold to generate the network
- Interactive Plotly network with zoom, pan, and adjustable similarity thresholds
- Node connectivity used to highlight semantic structure and interpret relationships learned by BERT

## 🧰 Technical Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **NLP Model**: `bert-base-uncased` from [Hugging Face Transformers](https://huggingface.co/bert-base-uncased)
- **Backend/Embedding Logic**: `transformers`, `torch`
- **Visualization**: `matplotlib`, `wordcloud`, `plotly`, `networkx`, `ipywidgets`


## 🗂️ Project Structure

```
├── app.py
├── bert_model.py
├── graph_app.py
├── README.md
├── requirements.txt
├── visualize_bert_word_similarity.ipynb
└── wordcloud_gen.py
```

## 📄 How It Works
- User inputs a word and selects how many similar words to display.
- The `BERTSimilarity` class computes BERT embeddings and returns top N similar words using cosine similarity.
- Results are displayed in four columns along with a word cloud visualization.

[LATEST]
- Built on with the full vocabulary (30,522 tokens) but excluding special tokens, subwords, non-alphabetic tokens & short words
- Accepts a word, top_n, and graph similarity threshold to generate the network
- Graph-based visualization using NetworkX
- Interactive Plotly network with zoom, pan, and adjustable similarity thresholds
- Node connectivity used to highlight semantic structure and interpret relationships learned by BERT

## 🛠️ Git Setup & Repository Cloning
If you haven't installed Git:

### 🔨 Install Git
**Windows:**

Download from https://git-scm.com/download/win and install with default settings.

**Ubuntu/Linux:**
```
sudo apt update
sudo apt install git
```

**macOS:**
```
brew install git
```

### 📦 Clone the Repository
```
git clone https://github.com/RohanSardar/bert_similarity_search.git
cd bert_similarity_search
```

## 🔧 Installation and Usage

Ensure you have the following installed:
- Python (= 3.10)
- Conda (for Conda-based setup)
- Virtualenv (install via `pip install virtualenv` if not already available)

### 🐍 Using conda
#### Create a conda virtual environment
Run the following command to create a virtual environment in a specific directory:
```
conda create -p venv python=3.10 -y
```
#### Activate it
```
conda activate venv/
```
#### Install dependencies
```
pip install -r requirements.txt
```

### 💻 Using virtualenv
Run the following command to create a virtual environment in a specific directory:
```
python -m virtualenv venv
```
#### Activate it
- **Windows**
```
venv\Scripts\activate
```
- **Linux/macOS**
```
source venv/bin/activate
```
#### Install dependencies
```
pip install -r requirements.txt
```

#### Finally run the app with:
```
streamlit run app.py
```
