import streamlit as st
import torch
import torch.nn.functional as F
import pickle
from model import MLPTextGenerator
import os

folder_path = os.path.dirname(__file__)
vocab_path = os.path.join(folder_path, "vocab.pkl")

@st.cache_resource
def load_model_and_vocab(model_choice, embed_dim, hidden_dim, context_size):
    with open("vocab_path", "rb") as f:
        data = pickle.load(f)

    word_to_idx = data["word_to_idx"]
    idx_to_word = data["idx_to_word"]
    vocab = data["vocab"]

    model_paths = {
        "Small": "holmes_small.pt",
        "Medium": "holmes_medium.pt",
        "Large": "holmes_large.pt"
    }

    model = MLPTextGenerator(len(vocab), embed_dim, hidden_dim, context_size)

    if model_choice in model_paths:
        model.load_state_dict(torch.load(model_paths[model_choice], map_location="cpu"))

    model.eval()
    return model, word_to_idx, idx_to_word, vocab


def predict_next_words(model, input_text, word_to_idx, idx_to_word,
                       k=5, temperature=1.0, context_size=5):

    tokens = input_text.lower().split()
    tokens = tokens[-context_size:]

    indices = [word_to_idx.get(w, word_to_idx.get("<unk>", 0)) for w in tokens]
    if len(indices) < context_size:
        indices = [word_to_idx.get("<unk>", 0)] * (context_size - len(indices)) + indices

    input_tensor = torch.tensor([indices], dtype=torch.long)

    generated = []
    for _ in range(k):
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs[0], 1).item()

        next_word = idx_to_word[next_idx]
        generated.append(next_word)
        indices = indices[1:] + [next_idx]
        input_tensor = torch.tensor([indices], dtype=torch.long)

    return " ".join(generated)


st.title("Sherlock Holmes Text Generator")

st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose model variant:",
    ["Small", "Medium", "Large", "Personalized"]
)

if model_choice == "Personalized":
    embed_dim = st.sidebar.slider("Embedding Dimension", 16, 256, 64, step=16)
    hidden_dim = st.sidebar.slider("Hidden Layer Size", 128, 2048, 1024, step=128)
    context_size = st.sidebar.slider("Context Size", 2, 10, 5)
    activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])
else:
    defaults = {
        "Small": (32, 512, 5),
        "Medium": (64, 1024, 5),
        "Large": (128, 2048, 5)
    }
    embed_dim, hidden_dim, context_size = defaults[model_choice]
    activation = "ReLU"

temperature = st.sidebar.slider("Temperature (controls randomness)", 0.1, 2.0, 1.0, step=0.1)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)
torch.manual_seed(random_seed)

model, word_to_idx, idx_to_word, vocab = load_model_and_vocab(
    model_choice, embed_dim, hidden_dim, context_size
)

st.subheader("Enter your prompt")
user_input = st.text_input("Type some text:", value="the adventure of")
k = st.slider("Number of words to generate", 1, 20, 5)

if st.button("Generate"):
    with st.spinner("Generating next words..."):
        output = predict_next_words(
            model, user_input, word_to_idx, idx_to_word,
            k=k, temperature=temperature, context_size=context_size
        )
    st.success("Generated text:")
    st.write(f"{user_input} {output}")

st.caption("Note: Unknown words are replaced with <unk>.")
