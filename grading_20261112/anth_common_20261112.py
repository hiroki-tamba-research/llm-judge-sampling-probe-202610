"""Shared helpers. Never prints key values."""
import os


def get_key(name):
    """Process env first, then the Windows per-user environment. Returns (value, source)."""
    v = os.environ.get(name)
    if v:
        return v, "process env"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            v, _ = winreg.QueryValueEx(k, name)
            if v:
                return v, "user env"
    except OSError:
        pass
    return None, "not set"


def list_models(client):
    return [m.id for m in client.models.list(limit=1000)]
