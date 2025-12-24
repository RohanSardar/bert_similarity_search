import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import torch
import torch.nn.functional as F
from bert_model import BERTSimilarity

@st.cache_resource
def load_bert_engine():
    return BERTSimilarity(vocab_size=30522) 

bert_model = load_bert_engine()

def get_sub_matrix(word_list, engine):
    inputs = engine.tokenizer(word_list, return_tensors="pt", padding=True, truncation=True).to(engine.device)
    with torch.no_grad():
        outputs = engine.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return torch.mm(embeddings, embeddings.t()).cpu().numpy()

st.set_page_config(page_title="BERT Similarity Network", page_icon="🕸️", layout='wide', initial_sidebar_state="expanded")
st.title("📋Interactive BERT Similarity Network🕸️")

st.sidebar.header("⚙️Configs")
seed_word = st.sidebar.text_input("Search Term", "science")
top_n = st.sidebar.slider("Number of Words", 10, 100, 30)
threshold = st.sidebar.slider("Connection Threshold", 0.8, 1.0, 0.85, 0.01)

if st.sidebar.button("Visualize"):

    similar_words = bert_model.get_similar_words(seed_word, top_n)
    st.sidebar.header("Similar Words:")

    for word, score in similar_words:
        st.sidebar.caption(f"{word} ({score:.3f})")

    with st.spinner(f"Finding words related to '{seed_word}'..."):
        try:
            similar_results = bert_model.get_similar_words(seed_word, top_n=top_n)
        except Exception as e:
            st.error(f"Error processing word: {e}")
            st.stop()
            
        found_words = [seed_word] + [item[0] for item in similar_results]
        
        sim_matrix = get_sub_matrix(found_words, bert_model)

        G = nx.Graph()
        
        for w in found_words:
            G.add_node(w)
            
        rows, cols = sim_matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                score = sim_matrix[i][j]
                if score > threshold:
                    G.add_edge(found_words[i], found_words[j], weight=score)

        pos = nx.spring_layout(G, seed=42, k=0.5) 

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x, node_y = [], []
        node_text = []
        node_adjacencies = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_adjacencies.append(len(G.adj[node]))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(showscale=True, colorscale='Plasma', reversescale=True, color=node_adjacencies,
                        size=15, colorbar=dict(thickness=15, title=dict(text='Connections', side='right'),
                                               xanchor='left')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"Network for '{seed_word}' (Similarity: {threshold})",
                            showlegend=False, hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        st.plotly_chart(fig, width='stretch')
        st.info("Use the 'Connection Threshold' slider to filter weak links. Zoom/Pan directly on the chart.", icon="ℹ️")