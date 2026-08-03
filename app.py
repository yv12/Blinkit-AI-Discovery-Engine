import gradio as gr
from api import app as fastapi_app, _load_state
import threading

# Load heavy data in the background so Uvicorn can bind to port 7860 instantly!
threading.Thread(target=_load_state, daemon=True).start()

# We MUST serve a real Gradio app at the root ("/") for Hugging Face to detect it
# and pass the health check. We embed our React app using a full-screen iframe!
css = """
footer { display: none !important; }
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; height: 100vh !important; }
"""

with gr.Blocks(css=css, title="Discovery Engine") as demo:
    gr.HTML('<iframe src="/react" style="width: 100%; height: 100vh; border: none;"></iframe>')

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
