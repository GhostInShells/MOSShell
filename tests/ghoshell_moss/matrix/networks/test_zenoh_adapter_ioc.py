"""A cell's channel runtime must resolve contracts from that cell's own Matrix container.

Protocol promise under test: whatever container the Matrix assembly hands to the network
adapter (`bind_ioc`) is the parent of every channel runtime container the adapter provides.
Otherwise a channel command's ``CommandUtil.force_get_contract`` sees an orphan container
and reports NOT_FOUND for contracts the cell itself registered.
"""
import logging
import os

import pytest
from ghoshell_container import Container

from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.core.blueprint.cell import Cell, NODE_ROLE
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.blueprint.project import NetworkMetadata
from ghoshell_moss.message import unique_id
from ghoshell_moss.matrix.networks.zenoh_adapter import ZenohAdapter


class _CellOwnedContract:
    """Stand-in for a contract the cell registers on its own Matrix container."""

    def __init__(self, marker: str = "cell-owned") -> None:
        self.marker = marker


@pytest.mark.asyncio
async def test_adapter_provided_channel_resolves_cell_matrix_contract():
    scope = unique_id()
    matrix_container = Container(name="matrix-of-this-cell")
    matrix_container.set(LoggerItf, logging.getLogger("test.zenoh_adapter_ioc"))
    matrix_container.set(_CellOwnedContract, _CellOwnedContract())

    cell = Cell(
        role=NODE_ROLE,
        name="probe_cell",
        home=os.getcwd(),
        persist=True,
    )

    adapter = ZenohAdapter(NetworkMetadata(scope=scope), is_host=False)
    # Matrix assembly order: sync bind_ioc -> sync bootstrap -> async adapter.__aenter__.
    adapter.bind_ioc(matrix_container)

    async with adapter:
        presence = adapter.new_presence(cell, logger=logging.getLogger("test.zenoh_adapter_ioc"))
        async with presence:
            provider = await presence.provide_channel(PyChannel(name='probe'))
            runtime_container = provider.container
            resolved = runtime_container.get(_CellOwnedContract)
            assert resolved is not None, (
                "channel runtime container cannot see the cell's own Matrix container; "
                f"parent chain is {runtime_container.bloodline!r}"
            )
            assert resolved.marker == "cell-owned"
