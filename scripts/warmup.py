from __future__ import annotations
import asyncio
from server.app import _startup, _shutdown


async def main():
    await _startup()
    print("Models loaded and warmed up.")
    await _shutdown()


if __name__ == "__main__":
    asyncio.run(main())