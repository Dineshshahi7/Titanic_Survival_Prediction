import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Define prediction function
def predict(feature1, feature2, feature3):
    # Convert to NumPy array (reshape for single sample)
    features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(features)
    return str(prediction[0])

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2"),
        gr.Number(label="Feature 3")
    ],
    outputs="text",
    title="Decision Tree Classifier",
    description="Enter feature values to get a prediction"
)

# Run app
if __name__ == "__main__":
    iface.launch()
