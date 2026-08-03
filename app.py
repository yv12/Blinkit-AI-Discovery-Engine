import gradio as gr
from api import app as fastapi_app

# Create a dummy Gradio interface
def greet():
    return "FastAPI is running!"

demo = gr.Interface(fn=greet, inputs=[], outputs="text")

# Mount the dummy Gradio app onto our existing FastAPI app.
# The Hugging Face Gradio SDK will detect this `app` variable 
# and use it to run our FastAPI server instead of a standard Gradio UI!
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
