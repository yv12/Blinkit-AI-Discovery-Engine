import gradio as gr
from api import app as fastapi_app, _load_state
import threading

# Load heavy data in the background so Uvicorn can bind to port 7860 instantly!
threading.Thread(target=_load_state, daemon=True).start()

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

app = gr.mount_gradio_app(fastapi_app, demo, path="/")
