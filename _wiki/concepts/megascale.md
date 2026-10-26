---
title: Megascale
type: concept
internal_only: false
tags: [tpu, training]
created: 2026-06-08
updated: 2026-10-03
---

## Overview and debugging

Megascale is the DCN collective communication library and runtime for
[[cloud-tpu]] training. A stalled collective is reported as a
[[megascale-hang]], but because Megascale sits in the middle, the root cause can
be customer software, Google compiler components, or TPU hardware.

## Connections

-   [[cloud-tpu]]
-   [[megascale-hang]]
