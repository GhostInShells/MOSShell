"""
Example NucleusFactory declarations for manifest discovery testing.

This file demonstrates the NucleusFactory interface pattern.
Place your own NucleusFactory instances here to have them
auto-discovered by `moss manifests nuclei`.
"""

from ghoshell_moss.core.blueprint.mindflow import (
    NucleusFactory,
    Nucleus,
    SignalName,
    SignalMeta,
    InputSignal,
)
from ghoshell_container import IoCContainer


class ExampleNucleusFactory(NucleusFactory):
    """A minimal example NucleusFactory for testing discovery."""

    def name(self) -> str:
        return "example_nucleus"

    def description(self) -> str:
        return "An example nucleus factory for manifest discovery testing"

    def signals(self) -> list[SignalMeta]:
        return [InputSignal]

    def factory(self, container: IoCContainer) -> Nucleus:
        raise NotImplementedError("Example stub — not intended for runtime use")


example_nucleus_factory = ExampleNucleusFactory()
