import os
import uvicorn
import spaces
from src.api.endpoint import app

@spaces.GPU
def dummy_gpu_function():
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
