import gradio as gr
from api import app as fastapi_app, _load_state
import threading

# Gradio's mount_gradio_app overwrites FastAPI startup events, so we must load manually.
# But loading the machine learning models synchronously blocks Uvicorn from starting,
# causing Hugging Face's health check to timeout and restart the container endlessly!
# By loading in a background thread, Uvicorn starts instantly and passes the health check.
threading.Thread(target=_load_state, daemon=True).start()

def greet():
    return "Health check passed!"

demo = gr.Interface(fn=greet, inputs=[], outputs="text")

# Mount Gradio at "/" so Hugging Face's health checker can find it.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
