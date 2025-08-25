"""
Version information for Data Integrity Agent
"""

__version__ = "1.0.0"
__author__ = "Popper Framework Team"
__description__ = "An agent for validating dataset quality and integrity"

def get_version():
    """Get the current version"""
    return __version__

def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__
    }
