import sys
import os
import time
import spaces
import gradio as gr

# Add backend directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.main import app as fastapi_app

# Define a dummy GPU function to satisfy Hugging Face ZeroGPU startup check
@spaces.GPU
def dummy_gpu_function():
    return "GPU active"

# Create a minimal Gradio blocks app to satisfy Gradio Space requirements
with gr.Blocks() as demo:
    gr.Markdown("# AtlasML API Server")
    gr.Markdown("FastAPI backend is running in the background.")

# Launch Gradio server in a non-blocking background thread (port 7860 is default)
demo.launch(prevent_thread_lock=True, server_name="0.0.0.0", server_port=7860)

# Mount our FastAPI backend onto Gradio's FastAPI server at root
# This routes all FastAPI endpoints (like /api/projects) successfully
demo.app.mount("/", fastapi_app)

if __name__ == "__main__":
    # Keep the main thread alive so the background Gradio/FastAPI server runs indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
