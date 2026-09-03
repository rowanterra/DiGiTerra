# Desktop entry point. Launcher logic in core/desktop.py. Usage: python desktop_app.py
import os

# A local user's own machine is not an attack surface. core.flask_app reads these
# when it is imported, so they are set before the import below.
os.environ.setdefault("DIGITERRA_MAX_UPLOAD_MB", "1024")
os.environ.setdefault("DIGITERRA_MAX_CELLS", "0")

from core.desktop import main  # noqa: E402

if __name__ == "__main__":
    main()
