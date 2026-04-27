"""
ghoshell CLI - Ghost In Shells command line tool
"""

from ghoshell_moss.cli.main import main, main_entry, app

# Import blueprint_cli to register its commands
from ghoshell_moss.cli import blueprint_cli

# Maintain backward compatibility, main variable is still available
__all__ = ['main', 'main_entry', 'app']
