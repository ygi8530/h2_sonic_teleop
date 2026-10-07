"""Shared interface to the H2 EDU policy server.

Imported by the Isaac Lab and MuJoCo simulation containers, which mount this
package read-only. It carries no simulator dependency: ``numpy`` and ``pyzmq``
only. :mod:`h2_policy_if.h2_joints` additionally needs nothing beyond the
standard library.
"""

from . import evaluation, h2_joints, protocol
from .client import H2PolicyClient, H2PolicyError
from .protocol import ProtocolError

__all__ = [
    "H2PolicyClient",
    "evaluation",
    "H2PolicyError",
    "ProtocolError",
    "h2_joints",
    "protocol",
]
