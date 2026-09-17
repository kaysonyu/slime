import ipaddress
import logging
import os
import random
import socket

logger = logging.getLogger(__name__)

SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):
        """
        Attempt to get the local LAN IP for the specific family (IPv4/IPv6).
        Strategy: UDP Probe (Preferred) -> Hostname Resolution (Fallback) -> None
        """

        # Strategy 1: UDP Connect Probe (Most accurate, relies on routing table)
        # Useful when the machine has a default gateway or internet access.
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                # The IP doesn't need to be reachable, but the routing table must exist.
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass  # Route unreachable or network error, move to next strategy.

        # Strategy 2: Hostname Resolution (Fallback for offline clusters)
        # Useful for offline environments where UDP connect fails but /etc/hosts is configured.
        try:
            # getaddrinfo allows specifying the family (AF_INET or AF_INET6)
            # Result format: [(family, type, proto, canonname, sockaddr), ...]
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]  # The first element of sockaddr is the IP
                # Must filter out loopback addresses to avoid "127.0.0.1" issues
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        # [Strict Mode] IPv6 Only
        # 1. Try UDP V6 Probe
        # 2. Try Hostname Resolution (V6)
        # If failed, fallback to V6 loopback. Never mix with V4.
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        # [Strict Mode] IPv4 Only (Default)
        # 1. Try UDP V4 Probe
        # 2. Try Hostname Resolution (V4)
        # If failed, fallback to V4 loopback. Never mix with V6.
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    """Wrap IPv6 address in [] if needed."""
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host
