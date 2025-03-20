import streamlit as st
import torch

st.title("Streamlit GPU Demo")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

if device.type == "cuda":
    st.write(f"GPU Name: {torch.cuda.get_device_name(0)}")
