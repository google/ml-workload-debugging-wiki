---
title: JAX
description: Pointer page. JAX internals are not covered here; see the backend pages.
type: entity
internal_only: false
tags: [training, pathways, tpu, gpu]
created: 2026-07-30
updated: 2026-10-15
---

# JAX

## Overview

This wiki does not document JAX itself.

## Backend details

The backend decides where the logs are:

-   Multi-controller: [[mcjax]]: pod naming, the two "worker zeros",
    `[process=0]`.
-   Single-controller: [[pathways]].

## Known issues

Version-specific JAX/libtpu defects: [[known-issues]].

## Connections

-   **Concepts**:
    -   [[known-issues]]
    -   [[pathways]]
-   **Entities**:
    -   [[libtpu]]
    -   [[mcjax]]
    -   [[xla]]
