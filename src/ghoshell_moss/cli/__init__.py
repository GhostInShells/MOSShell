"""
ghoshell CLI - Ghost In Shells command line tool
"""

from ghoshell_moss.cli.main import main, main_entry, app

# Maintain backward compatibility, main variable is still available
__all__ = ['main', 'main_entry']
