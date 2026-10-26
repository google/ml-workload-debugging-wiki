---
title: Network and TPU Stacks
type: concept
internal_only: false
tags: [tpu, training, datacenter]
created: 2026-05-13
updated: 2026-10-03
---

## Definition

TPU workloads rely on a complex stack of software and hardware for execution and
communication.

### Software Stack

-   **Libtpu**: A user-space shared library (`libtpu.so`) that provides
    compilers (XLA), runtimes (TFRT, PJRT), a hardware abstraction layer (HAL)
    and the userspace driver (USD). It ships no kernel driver: the USD reaches
    the TPU through the VM kernel's VFIO subsystem (`/dev/vfio`), which is part
    of the OS image. It allows user applications to utilize TPU resources from
    VMs.
-   **GCE hypervisor**: GCP's user-mode hypervisor on top of KVM. It provides
    virtualization services and allows hosting the guest OS with XLA compilers,
    TPU runtime, TPU drivers, and user applications. It drives KVM and
    implements virtual devices via device-passthrough.

### Networking

-   **Inter-Chip Interconnect (ICI)**: High-speed interconnect linking TPUs
    within a pod/slice. Communication within a slice happens via ICI.
-   **Data Center Network (DCN)**: Standard network used for communication
    across different slices or with external systems. Communication across
    multiple slices in a multi-slice workload uses DCN.

## Why it matters for debugging

Hangs and performance issues can occur at various levels of the stack:

-   **Libtpu/XLA level**: Compiler bugs, inconsistent programs.
-   **Hypervisor (GCE hypervisor) level**: Virtualization issues, device passthrough
    failures.
-   **Network level**: ICI link failures, DCN congestion or drops.

## Common issues related to it

-   **Communication Hangs**: Often caused by dropped packets or link failures in
    ICI or DCN.
-   **Passthrough Failures**: Issues with mounting VFIO device nodes in the VM.
