"""MOSS debug node: dump all IoC container contracts and providers on start.

Start:  moss nodes run .moss/system_test_nodes/contracts_dump
Debug:  python main.py
"""

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    container = matrix.container

    print("=== Bound Contracts ===")
    for contract in sorted(container.contracts(), key=str):
        bound = container.get_bound(contract)
        label = _describe_bound(bound)
        print(f"  {contract.__name__:<40} -> {label}")

    print()
    print("=== Registered Providers ===")
    for provider in container.providers():
        contract = provider.contract()
        contract_name = getattr(contract, '__name__', str(contract))
        provider_type = type(provider).__name__
        singleton = "singleton" if provider.singleton() else "factory"
        print(f"  {contract_name:<40} <- {provider_type} ({singleton})")


def _describe_bound(bound) -> str:
    if bound is None:
        return "None"
    if hasattr(bound, 'contract') and callable(bound.contract):
        return f"Provider<{type(bound).__name__}>"
    return type(bound).__name__


if __name__ == "__main__":
    Matrix.discover().run(main)
