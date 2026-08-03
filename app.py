import gradio as gr
import spaces
from api import app as fastapi_app, _load_state
import threading

import sys
import traceback
import logging

logger = logging.getLogger("api")

def _safe_load():
    try:
        _load_state()
        logger.info("STATE LOADED OK")
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        sys.stdout.flush()

# Load heavy data in the background so Uvicorn can bind to port 7860 instantly!
threading.Thread(target=_safe_load, daemon=True).start()

# Hugging Face ZeroGPU environments strictly require at least one function
# decorated with @spaces.GPU, otherwise they crash the container on startup.
@spaces.GPU
def dummy_gpu_function():
    pass


# We MUST serve a real Gradio app at the root ("/") for Hugging Face to detect it
# and pass the health check. We embed our React app using a full-screen iframe!
# We inject a <style> block directly into the HTML to bypass Gradio 6.0 css restrictions.
html_content = """
<style>
footer { display: none !important; }
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; height: 100vh !important; }
</style>
<iframe src="/react" style="width: 100%; height: 100vh; border: none;"></iframe>
"""

with gr.Blocks(title="Discovery Engine") as demo:
    gr.HTML(html_content)
    
    # Bind the dummy function to a hidden button so Gradio's internal API registers it.
    # Without this, ZeroGPU ignores the function and crashes!
    with gr.Row(visible=False):
        hidden_btn = gr.Button("Init ZeroGPU")
        hidden_btn.click(fn=dummy_gpu_function, inputs=[], outputs=[])

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
