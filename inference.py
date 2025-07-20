import streamlit as st

st.set_page_config(page_title="L&T Building Classifier", layout="centered")

import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import requests

# -----------------------
# Load Model and Classes
# -----------------------

@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    checkpoint = torch.load("class.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    class_names = checkpoint['class_names']
    normalization_stats = checkpoint['normalization']
    model.eval()
    return model, class_names, normalization_stats

model, class_names, normalization_stats = load_model()
class_names = ['CRR Centre', 'EDRC', 'Technology Centre-2', 'Technology Centre-3']

# -----------------------
# Transforms
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# Building Metadata
# -----------------------

class_info = {
} #your information

# -----------------------
# Helper Functions
# -----------------------

def predict_building(image, confidence_threshold=0.7):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)

    confidence = max_prob.item()

    if confidence < confidence_threshold:
        return "unknown", confidence
    else:
        return class_names[pred_idx.item()], confidence

def generate_prompt(building, user_question):
    info = building_info[building]
    return f"""
You are a helpful assistant answering queries about L&T buildings in Chennai. Answer the user's question using ONLY the facts below. Do NOT add any extra information or commentary.

Building: {building}
Alias: {info['alias']}
Location: {info['location']}
Structure: {info['structure']}
Awards: {info['awards']}
Features:
{chr(10).join('- ' + feature for feature in info['features'])}
Notable: {info['notable']}

Answer the following user question based on the building info:
{user_question}
### Response Rules
1. Answer concisely in 1-3 sentences
2. Use ONLY the above facts
3. If information isn't in facts, say "I don't have that information"
4. Never ask follow-up questions
"""

def query_ollama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        return res.json().get("response", "No response from model.")
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# -----------------------
# Streamlit UI
# -----------------------

st.title("L&T Building Identifier + Assistant")

uploaded_image = st.file_uploader("Upload a photo of the building", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predicted_building, confidence = predict_building(image)
    if predicted_building == "unknown":
        st.warning(f"This building is unknown (confidence: {confidence:.2f})")
    else:
        st.success(f"This looks like: **{predicted_building}** (confidence: {confidence:.2f})")

    question = st.text_input("Ask a question about this building:")
    if question:
        prompt = generate_prompt(predicted_building, question)
        with st.spinner("Thinking with LLaMA..."):
            response = query_ollama(prompt)
        st.markdown("**LLaMA's Response:**")
        st.write(response)
