import uvicorn
from app.api import make_app

if __name__ == "__main__":
    uvicorn.run(make_app(), host="0.0.0.0", port=8000, log_level="info")
