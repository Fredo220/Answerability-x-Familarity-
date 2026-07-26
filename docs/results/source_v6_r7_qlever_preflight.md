# Source-v6 R7 QLever Preflight

**Status:** Blocked before construction

The Git-bound R7 commit was
`32d42d9dfa6227e43ce13107a8087cfbc2d6706c`.
The registered `LIMIT 1` syntax-only preflight started with the Person query.
The QLever gateway returned HTTP 502 after approximately 30 seconds. One exact
transport retry returned the same HTTP 502. No binding row, candidate identity,
rank, domain yield, Gemma output, or research endpoint was retained or
inspected.

The failure is attributed to the cost of the global case-insensitive label and
alternative-label scans added in R7. Under the R7 stop rule, R7 is blocked and
cannot be repaired in place. It provides no evidence for or against
Familiarity-by-Answerability.
