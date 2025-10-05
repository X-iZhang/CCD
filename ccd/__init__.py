try:
    from .run_ccd import ccd_eval
    from .run_ccd import run_eval
except ImportError as e:
    import warnings
    warnings.warn(f"⚠️ CCD loaded without ccd_eval: {e}")