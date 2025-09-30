import sys
import requests

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client/upload_test.py /path/to/audio.wav")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "rb") as f:
        files = {"file": (path.split("/")[-1], f, "application/octet-stream")}
        r = requests.post("http://localhost:8000/transcribe", files=files)
        print(r.json())
