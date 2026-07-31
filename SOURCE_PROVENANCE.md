# Source provenance

This public code release was assembled from the governed NeuroGenis Paper-02
record on 2026-07-31.

| Object | Provenance |
| --- | --- |
| Public v1 repository tip | `a8d6622` |
| Governed PoC source commit | `abc68975322c233711e1c427246921fbeb46b51d` |
| Revised arXiv source archive SHA-256 | `94b20bfa4bbd64e41229eb9389f813c5a6f2840a82533c02004ef3a5f8dc5de5` |
| `src/lrtc.py` source commit | `9e6a7196d744d26eafcd460085ca6fb2921306cb` |
| `src/lrtc.py` blob SHA-256 | `ff459454e346d7863dd1ef30da6da218100b80446e66a3b3a9138c24b6f380aa` |

The governed Paper-02 scripts on the PoC main branch imported `src.lrtc`, but
that module was missing from the cleaned main tree. The exact implementation
and its estimator tests remained preserved at commit `9e6a719`. This release
restores that preserved implementation rather than recreating it from the
manuscript description.

Generated result files are intentionally not copied into this repository.
Their absence is a data-governance choice, not evidence that the public code was
run without the restricted inputs.
