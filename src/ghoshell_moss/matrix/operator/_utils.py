"""
Service operator key expression toolkit — zero magic strings.

Mirrors matrix/networks/_utils.py pattern:
  ServiceKeyspace = namespace-scoped key builders + wildcards
  ServiceKeyExpr  = per-service key pack (constructed once per provide() call)
"""

from typing import ClassVar

from ghoshell_moss.core.blueprint.cell import CellAddress

__all__ = ['ServiceKeyspace', 'ServiceKeyExpr', '_META_KEY']

_META_KEY: ClassVar[str] = 'meta'
"""Reserved business key for the auto-declared meta queryable.

The discovery pipeline depends on this: ``get_services_by_kind`` queries
``query/{_META_KEY}`` on every live service.  Do not rename without
updating all consumers.
"""


class ServiceKeyspace:
    """Service namespace key packaging (namespace-scope).

    All wildcards and key parsers live here.  One instance per ZenohOperator.
    """

    SLOTS: ClassVar[tuple[str, ...]] = ('pub', 'listen', 'query', 'live')

    def __init__(self, network_ns: str):
        self.services_ns = f"{network_ns}/services"
        self.services_ns_prefix = self.services_ns + '/'

        # wildcards for discovery
        self.all_live_wildcard = f"{self.services_ns}/**"

    def per_service(self, address: CellAddress, kind: str) -> 'ServiceKeyExpr':
        return ServiceKeyExpr(self, address, kind)

    # -- wildcards -------------------------------------------------------

    def kind_live_wildcard(self, kind: str) -> str:
        return f"{self.services_ns}/**/{kind}/live"

    def kind_query_meta_wildcard(self, kind: str) -> str:
        return f"{self.services_ns}/**/{kind}/query/{_META_KEY}"

    # -- wildcards for client-side operations ------------------------------

    def kind_pub_wildcard(self, kind: str, key: str) -> str:
        """Wildcard for subscribing to all services of a kind on a pub key."""
        return f"{self.services_ns}/**/{kind}/pub/{key}"

    # -- key parsers -------------------------------------------------------

    def parse_key(self, key: str) -> tuple[str, str, str, str] | None:
        """Parse a full service key → (dotted_addr, kind, slot, rest).

        ``rest`` is the business key suffix (may contain '/' for sub-keys).
        Returns None if the key does not start with ``services_ns_prefix``
        or has fewer than 4 segments after the prefix.

        Example: ``{ns}/global.host.main/webview/pub/badge``
              → ``('global.host.main', 'webview', 'pub', 'badge')``
        """
        if not key.startswith(self.services_ns_prefix):
            return None
        rest = key[len(self.services_ns_prefix):]
        parts = rest.split('/', 3)
        if len(parts) < 4:
            return None
        return parts[0], parts[1], parts[2], parts[3]

    def parse_live_identity(self, identity: str) -> tuple[str, str] | None:
        """Extract (dotted_addr, kind) from a liveness identity.

        Identity is the suffix after services_ns_prefix, e.g.
        ``global.host.main/webview/live``.
        """
        if not identity.endswith('/live'):
            return None
        inner = identity[:-len('/live')]  # "global.host.main/webview"
        parts = inner.rsplit('/', 1)
        if len(parts) != 2:
            return None
        return parts[0], parts[1]


class ServiceKeyExpr:
    """Per-service (address + kind) key expressions.

    Constructed once per ``provide()`` call, read-only thereafter.
    All zenoh key strings flow through this object — no f-strings in operator code.
    """

    def __init__(self, keyspace: ServiceKeyspace, address: CellAddress, kind: str):
        dotted = address.replace('/', '.')
        base = f"{keyspace.services_ns}/{dotted}/{kind}"

        self.address = address
        self.kind = kind
        self.dotted = dotted
        self.base = base

        # slot prefixes
        self.query_prefix = f"{base}/query/"
        self.pub_prefix = f"{base}/pub/"
        self.listen_prefix = f"{base}/listen/"

        # liveness token key (no business key suffix)
        self.live_key = f"{base}/live"

    # -- per-business-key builders ---------------------------------------

    def query_key(self, business_key: str) -> str:
        return self.query_prefix + business_key

    def pub_key(self, business_key: str) -> str:
        return self.pub_prefix + business_key

    def listen_key(self, business_key: str) -> str:
        return self.listen_prefix + business_key

    def meta_query_key(self) -> str:
        return self.query_key(_META_KEY)
