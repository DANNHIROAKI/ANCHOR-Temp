# Canonical workload format v1

Performance inputs use an uncompressed, little-endian, structure-of-arrays
binary format. The layout is intentionally simple: preprocessing owns all
coordinate conversion, while the benchmark process can read each array once
into its final anonymous allocation.

## Header

The header is exactly 128 bytes. Integer fields are unsigned and little-endian.

| Offset | Type | Field |
|---:|---|---|
| 0 | `char[8]` | `ANCHORW\0` |
| 8 | `uint16` | format version (`1`) |
| 10 | `uint8` | byte order (`1` = little-endian) |
| 11 | `uint8` | endpoint type (`1` = binary64, `2` = int64) |
| 12 | `uint32` | dimension |
| 16 | `uint32` | flags (must be zero) |
| 20 | `uint32` | header size (`128`) |
| 24 | `uint64` | number of R objects |
| 32 | `uint64` | number of S objects |
| 40 | `uint64` | R-id array offset |
| 48 | `uint64` | R-lower array offset |
| 56 | `uint64` | R-upper array offset |
| 64 | `uint64` | S-id array offset |
| 72 | `uint64` | S-lower array offset |
| 80 | `uint64` | S-upper array offset |
| 88 | `uint64` | total file size |
| 96 | `byte[32]` | SHA-256 of the six logical arrays |

Every array starts on a 64-byte boundary; padding bytes are zero. IDs are
`uint64`. Lower and upper endpoint arrays have row-major shape `(n, d)`. The
logical digest concatenates the six arrays in header order and excludes
padding. A separate manifest records the SHA-256 of the complete file.

Canonical workloads contain only positive, finite boxes. Empty boxes may occur
in a general algorithm API, but dataset construction filters or rejects them
before freezing a performance workload.

## Manifest

`<workload>.manifest.json` uses schema `anchor-workload-manifest-v1`. It records
the file and payload digests, dimensions, endpoint type, object counts, ID-list
digests, coordinate ranges, and dataset-specific provenance under `metadata`.
The JSON encoding is `anchor-canonical-json-v1`, defined in
`anchor_exp.stable_hash`.

The reader validates header bounds, alignment, logical payload SHA-256,
endpoint finiteness, and positive side lengths before exposing arrays.
