# Upstream drafts. NOT POSTED.

These are drafts of a comment on NIR issue #208 and of bug reports for each project. Nothing here has been posted. Read each one, re-run its reproduction on the latest release, and search the project's open issues for duplicates before posting.

| File | Target | Contents |
|---|---|---|
| `nir-208-comment.md` | neuromorphs/NIR #208 | Proposal for shared fixtures, plus a summary of what the prototype found |
| `nir-issues.md` | neuromorphs/NIR | Grouped-conv `input_type` (NIR-1); `v_reset` forward-compatibility (NIR-2) |
| `nirtorch-issues.md` | neuromorphs/NIRTorch | Legacy loader: fan-in order (NT-1), silent node drop (NT-2), hash-seed dependence (NT-3) |
| `snntorch-issues.md` | jeshraghian/snntorch | ST-1 to ST-5 |
| `norse-issues.md` | norse/norse | NO-1 to NO-3, NO-6, NO-7 |
| `sinabs-issues.md` | synsense/sinabs | SB-1 to SB-3, SB-5, SB-6 |
| `rockpool-issues.md` | SynSense Rockpool | RP-1 to RP-3 |
| `spyx-issues.md` | kmheckel/spyx | SX-2 to SX-5 |
| `repros/*.py` | — | Minimal standalone reproductions (each needs only `nir` plus that framework). All were run on 2026-11-16 and show the bug |

Suggested order: post the #208 comment first and agree on a fixture format with the maintainers. Then file NIR-2 (it blocks Sinabs and Rockpool), NT-1/NT-2 (silent wrong results in three frameworks), RP-1, and ST-2.
