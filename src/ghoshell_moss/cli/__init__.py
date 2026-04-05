"""
ghoshell CLI - Ghost In Shells command line tool
"""

from ghoshell_moss.cli.main import main, main_entry, app

# Maintain backward compatibility, main variable is still available
__all__ = ['main', 'main_entry']

# Auto-import all command modules
import ghoshell_moss.cli.codex
import ghoshell_moss.cli.concepts

# import ghoshell_moss.cli.blueprint
# import ghoshell_moss.cli.inspect
#
app.add_typer(codex.app, name="codex")
app.command(name='concepts')(concepts.show_concepts)
