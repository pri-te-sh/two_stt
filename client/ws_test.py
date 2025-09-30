import sys
import time
import websockets

async def main(path: str):
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri, max_size=None) as ws:
        # Stream file in chunks
        with open(path, "rb") as f:
            while True:
                chunk = f.read(32000)  # ~20ms @ 16k mono float32, but it's just bytes here
                if not chunk:
                    break
                await ws.send(chunk)
                await asyncio.sleep(0.05)
        await ws.send("DONE")
        async for msg in ws:
            print(msg)

if __name__ == "__main__":
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python client/ws_test.py /path/to/audio.wav")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))