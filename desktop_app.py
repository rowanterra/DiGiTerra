"""
DiGiTerra desktop launcher. Starts the Flask app in a background thread and opens
a pywebview window. Handoff note: Save-file dialogs go through the js_api (DesktopApi);
the front end calls into that when running in the desktop build. See HANDOFF.md.
"""
import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
import shutil
import platform

import webview

from app import create_app

# Platform-specific paths
if platform.system() == "Windows":
    # Windows: Use AppData directories
    LOG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra" / "Logs"
    APP_SUPPORT_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra"
elif platform.system() == "Linux":
    # Linux: Use XDG directories
    xdg_data_home = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    APP_SUPPORT_DIR = Path(xdg_data_home) / "DiGiTerra"
    LOG_DIR = Path(xdg_cache_home) / "DiGiTerra" / "logs"
else:
    # macOS: Use Library directories
    LOG_DIR = Path.home() / "Library" / "Logs" / "DiGiTerra"
    APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "DiGiTerra"

LOG_FILE = LOG_DIR / "digiterra.log"


class DesktopApi:
    def __init__(self, output_dir: Path, window=None) -> None:
        self.output_dir = output_dir
        self.window = window
        # Guard to prevent multiple simultaneous save dialogs, which on macOS
        # can manifest as many app icons / windows being spawned.
        self._saving = False

    def test_api(self) -> str:
        """Test method to verify API is accessible from JavaScript."""
        try:
            # Simple test that doesn't access any window or webview state
            return "API is working"
        except Exception as error:
            logging.error("Error in test_api: %s", error)
            # Return a safe value instead of raising
            return "API error"

    def save_file(self, filename: str, download_name: str | None = None) -> bool:
        # Simple re-entrancy guard: if a save dialog is already open/processing,
        # ignore additional requests. This prevents rapid repeated JS calls from
        # opening many native save panels (and duplicate Dock icons on macOS).
        if self._saving:
            logging.info("save_file called while another save is in progress; ignoring")
            return False

        self._saving = True
        try:
            source_path = self.output_dir / filename
            if not source_path.exists():
                logging.error("Download source not found: %s", source_path)
                return False

            # Get window reference - try stored reference, then webview.windows
            window = self.window
            if window is None:
                try:
                    if webview.windows:
                        window = webview.windows[0]
                        self.window = window  # Cache it for future use
                except (AttributeError, IndexError) as e:
                    logging.warning("Could not access webview.windows: %s", e)
            
            # Try using window instance first (more reliable in compiled apps)
            result = None
            if window:
                try:
                    result = window.create_file_dialog(
                        webview.SAVE_DIALOG,
                        save_filename=download_name or filename,
                    )
                    logging.debug("File dialog opened using window instance")
                except Exception as e:
                    logging.warning("Window-based file dialog failed, trying module-level: %s", e)
            
            # Fallback to module-level function
            if result is None:
                try:
                    result = webview.create_file_dialog(
                        webview.SAVE_DIALOG,
                        save_filename=download_name or filename,
                    )
                    logging.debug("File dialog opened using module-level function")
                except Exception as e:
                    logging.error("Both file dialog methods failed: %s", e)
                    return False

            if not result:
                logging.info("User cancelled file save dialog")
                return False

            destination = result[0] if isinstance(result, (list, tuple)) else result
            shutil.copyfile(source_path, destination)
            logging.info("File saved successfully: %s -> %s", source_path, destination)
            return True
        except Exception as error:
            _log_exception("Failed to save file", error)
            return False
        finally:
            # Always release the guard so future saves work
            self._saving = False


def _configure_logging(debug: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )


def _log_exception(context: str, error: BaseException) -> None:
    logging.error("%s: %s", context, error)
    logging.error("Traceback:\n%s", "".join(traceback.format_exception(error)))


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(host, port, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _start_server(host, port):
    try:
        app = create_app()
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as error:
        _log_exception("Server failed to start", error)


def _show_error_window(message: str) -> None:
    html = f"""
    <html>
      <head><title>DiGiTerra Error</title></head>
      <body style="font-family: -apple-system, Helvetica, Arial, sans-serif; padding: 24px;">
        <h2>DiGiTerra could not start</h2>
        <p>{message}</p>
        <p>Check the log file at <code>{LOG_FILE}</code> for details.</p>
      </body>
    </html>
    """
    webview.create_window("DiGiTerra Error", html=html)
    webview.start()


def main():
    debug = os.environ.get("DIGITERRA_DEBUG") == "1"
    _configure_logging(debug)

    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    os.environ["DIGITERRA_BASE_DIR"] = str(base_dir)
    os.chdir(base_dir)
    output_dir = Path(
        os.environ.get(
            "DIGITERRA_OUTPUT_DIR",
            APP_SUPPORT_DIR / "user_visualizations",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DIGITERRA_OUTPUT_DIR"] = str(output_dir)

    host = "127.0.0.1"
    port_env = os.environ.get("DIGITERRA_PORT")
    port = int(port_env) if port_env else _find_free_port()
    
    # Create API instance first (window will be set after window creation)
    api = DesktopApi(output_dir)

    try:
        server_thread = threading.Thread(
            target=_start_server,
            args=(host, port),
            daemon=True,
            name="digiterra-server",
        )
        server_thread.start()

        if not _wait_for_server(host, port):
            raise RuntimeError("Failed to start DiGiTerra server.")

        # Create window and store reference in API
        # Note: js_api must be passed during window creation for pywebview to expose it
        # Use resizable=True to allow window resizing
        window = webview.create_window(
            "DiGiTerra",
            f"http://{host}:{port}",
            js_api=api,
            min_size=(1024, 768),
            resizable=True,
        )
        api.window = window  # Store window reference for file dialogs
        # Start webview - this blocks until window is closed
        webview.start(debug=debug)
    except Exception as error:
        _log_exception("Desktop app failed to launch", error)
        _show_error_window("The app encountered an unexpected error while launching.")


if __name__ == "__main__":
    main()
