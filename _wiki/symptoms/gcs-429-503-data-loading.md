---
title: GCS 429/503 Errors during Data Loading
type: symptom
internal_only: false
tags: [tpu, training]
created: 2026-06-13
updated: 2026-10-03
---

## Symptom signature

Startup or data loading stalls at thousands of chips; app logs show GCS reads
failing with:

```
google.api_core.exceptions.ServiceUnavailable: 503 GCS Service Unavailable
...
google.api_core.exceptions.TooManyRequests: 429 Rate limit exceeded for bucket <bucket-name>
```

or transport errors during GCS reads. Seen at 32k chips: thousands of TPU VMs
read one bucket at once and exceed its request-rate or egress limits.

## Triage and Mitigation

1.  Enable GCS Anywhere Cache to take read load off the bucket.
2.  Use a Hierarchical Namespace (HNS) bucket for higher request rates.
3.  Fewer, larger reads; stagger data-loading start across workers. Local
    caching on ephemeral disk can trigger [[stopiteration-disk-full]].

## Connections

-   **Symptoms**:
    -   [[stopiteration-disk-full]]
-   **Entities**:
    -   [[cloud-storage]]
