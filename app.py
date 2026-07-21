import sys
import os
import spaces

# Add backend directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import uvicorn
from app.main import app

@spaces.GPU
def dummy_gpu_function():
    # This dummy function is required to satisfy Hugging Face ZeroGPU startup validation
    return "GPU active"

if __name__ == "__main__":
    # Hugging Face routes incoming traffic on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
