"""Start the server: `py -m padel.server`, then open the printed address on a phone on the same Wi-Fi."""
import socket

import uvicorn

from .app import create_app

PORT = 8000


def lan_ip() -> str:
    """This laptop's address on the local network (connecting a UDP socket sends nothing)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    print(f"Open http://{lan_ip()}:{PORT} on your phone (same Wi-Fi), or http://localhost:{PORT} here.")
    uvicorn.run(create_app(), host="0.0.0.0", port=PORT)
