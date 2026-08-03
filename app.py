import gradio as gr
from api import app as fastapi_app, _load_state

# Gradio's mount_gradio_app overrides FastAPI's @app.on_event("startup"),
# so we must manually load the data into memory before mounting.
_load_state()

def greet():
    return "Health check passed!"

demo = gr.Interface(fn=greet, inputs=[], outputs="text")

# Mount Gradio at "/" so Hugging Face's health checker can find it.
# Because api.py already defines @app.get("/"), that exact route will
# take precedence and serve your React app, while Gradio handles /info!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
