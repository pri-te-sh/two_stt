"""Simple test client for STT WebSocket server."""
import asyncio
import json
import wave
import numpy as np
from websockets import connect
from pathlib import Path

async def test_client(audio_file: str, server_url: str = "ws://localhost:8081/ws"):
    """
    Test client that streams audio file to STT server.
    
    Args:
        audio_file: Path to WAV file (must be mono, 16kHz, PCM16)
        server_url: WebSocket server URL
    """
    print(f"Connecting to {server_url}...")
    
    async with connect(server_url) as websocket:
        print("Connected!")
        
        # Send start message
        start_msg = {"event": "start", "language": "auto"}
        await websocket.send(json.dumps(start_msg))
        print(f"Sent start message: {start_msg}")
        
        # Load audio file
        print(f"Loading audio file: {audio_file}")
        with wave.open(audio_file, 'rb') as wf:
            # Verify format
            assert wf.getnchannels() == 1, "Audio must be mono"
            assert wf.getsampwidth() == 2, "Audio must be 16-bit"
            assert wf.getframerate() == 16000, "Audio must be 16kHz"
            
            # Read all frames
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        print(f"Loaded {len(audio_array)} samples ({len(audio_array)/16000:.2f}s)")
        
        # Stream audio in chunks
        chunk_size = 1600  # 100ms at 16kHz
        num_chunks = len(audio_array) // chunk_size
        
        # Start receiving task
        receive_task = asyncio.create_task(receive_messages(websocket))
        
        print("Streaming audio...")
        for i in range(num_chunks):
            chunk = audio_array[i*chunk_size:(i+1)*chunk_size]
            await websocket.send(chunk.tobytes())
            await asyncio.sleep(0.1)  # Real-time streaming
        
        # Send remaining samples
        if len(audio_array) % chunk_size > 0:
            remaining = audio_array[-(len(audio_array) % chunk_size):]
            await websocket.send(remaining.tobytes())
        
        print("Audio streaming complete")
        
        # Send stop to force finalize
        await asyncio.sleep(1.0)  # Wait for final transcription
        stop_msg = {"event": "stop"}
        await websocket.send(json.dumps(stop_msg))
        print(f"Sent stop message: {stop_msg}")
        
        # Wait a bit for final results
        await asyncio.sleep(2.0)
        
        # Cancel receive task
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
        print("Test complete!")


async def receive_messages(websocket):
    """Receive and print messages from server."""
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "interim":
                    print(f"[INTERIM] {data['text']}")
                elif msg_type == "final":
                    print(f"[FINAL] {data['text']}")
                    if 'language' in data and data['language']:
                        print(f"  Language: {data['language']}")
                elif msg_type == "status":
                    print(f"[STATUS] Backpressure: {data['backpressure']}, "
                          f"Cooldown: {data['cooldown_ms']}ms, "
                          f"Tail: {data['tail_s']}s")
                elif msg_type == "error":
                    print(f"[ERROR] {data['code']}: {data['detail']}")
                else:
                    print(f"[UNKNOWN] {data}")
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <audio_file.wav> [server_url]")
        print("Audio file must be: mono, 16kHz, PCM16")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8081/ws"
    
    asyncio.run(test_client(audio_file, server_url))
