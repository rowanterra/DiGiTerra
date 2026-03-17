"""Flask blueprints for DiGiTerra. Register in app.py with register_blueprints(app)."""
from python_scripts import config

from .main import bp as main_bp
from .upload import bp as upload_bp
from .exploration import bp as exploration_bp
from .preprocess import bp as preprocess_bp
from .modeling import bp as modeling_bp
from .prediction import bp as prediction_bp
from .assets import bp as assets_bp


def register_blueprints(app):
    """Register all blueprints. When URL_PREFIX is set, register at both root and prefix so unprefixed and prefixed routes both work."""
    prefix = config.URL_PREFIX or None
    blueprints = [main_bp, upload_bp, exploration_bp, preprocess_bp, modeling_bp, prediction_bp, assets_bp]
    for bp in blueprints:
        app.register_blueprint(bp, url_prefix=None)
        if prefix:
            app.register_blueprint(bp, url_prefix=prefix, name=f"{bp.name}_prefixed")
