# Desktop entry point. Launcher logic in core/desktop.py. Usage: python desktop_app.py
import os

# One user on one machine, so the limits are those of the machine rather than of a
# shared server: a 1 GB body, 200 million cells, a 4 GB frame, and no total across
# sessions (there is one session). A file past these is refused before parsing.
# core.flask_app reads these when it is imported, so they are set before the import.
os.environ.setdefault("DIGITERRA_MAX_UPLOAD_MB", "1024")
os.environ.setdefault("DIGITERRA_MAX_CELLS", "200000000")
os.environ.setdefault("DIGITERRA_MAX_FRAME_MB", "4096")
os.environ.setdefault("DIGITERRA_MAX_TOTAL_FRAME_MB", "0")
# Bundle import unpickles fitted objects, which runs code from the file. It is on
# here because every import is checked against this installation's signing key
# first: a bundle from elsewhere is refused unless its key is shared deliberately.
os.environ.setdefault("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")
# The data-check override is a desktop-only control; the served build has none.
os.environ.setdefault("DIGITERRA_ALLOW_UNSAFE_OVERRIDE", "1")

from core.desktop import main  # noqa: E402

if __name__ == "__main__":
    main()
