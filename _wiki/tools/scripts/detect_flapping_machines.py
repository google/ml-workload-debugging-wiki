#!/usr/bin/env python3
"""Detects flapping TPU machines and mass events from GCE system events."""

import argparse
import datetime
import json
import os
import sys
from typing import Any

_ROLLING_WINDOW_SEC = 86400
_OUTAGE_MERGE_WINDOW_SEC = 900
_STRIKE_COOLDOWN_SEC = 7200


def _parse_iso_ts(ts_str: str) -> datetime.datetime | None:
  """Parses an ISO-8601 timestamp string into a UTC-aware datetime.

  Args:
    ts_str: ISO-8601 timestamp string.

  Returns:
    A UTC-aware datetime, or None if ts_str is empty or cannot be parsed.
  """
  if not ts_str:
    return None
  s = ts_str.strip()
  if s.endswith("Z"):
    s = s[:-1] + "+00:00"
  try:
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt
  except ValueError:
    return None


def _load_mapping(
    mapping_file: str, notes: list[str]
) -> dict[str, list[dict[str, Any]]]:
  """Loads GCE instance_id to cluster machine mapping rows from JSONL."""
  mapping: dict[str, list[dict[str, Any]]] = {}
  if not mapping_file or not os.path.isfile(mapping_file):
    notes.append(
        "No mapping_file provided or found; falling back to GCE instance_id "
        "keying, which undercounts outages and strikes across VM/MIG recreates."
    )
    return mapping

  multi_host_ids: set[str] = set()
  with open(mapping_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      inst_id = str(row.get("instance_id", "")).strip()
      if not inst_id:
        continue
      mapping.setdefault(inst_id, []).append(row)
      if int(row.get("n_hosts_2d", 1) or 1) > 1:
        multi_host_ids.add(inst_id)

  for inst_id, rows in mapping.items():
    machines = {
        row.get("host_id") for row in rows if row.get("host_id")
    }
    if len(machines) > 1:
      notes.append(
          f"instance_id {inst_id} maps to multiple cluster machines "
          f"({sorted(machines)}); resolving per event by nearest last_seen."
      )
  if multi_host_ids:
    notes.append(
        f"instance_ids with n_hosts_2d > 1 ({sorted(multi_host_ids)}) ran on "
        "multiple machines in their table window; events may span machines."
    )
  return mapping


def _resolve_host_machine(
    inst_id: str,
    event_dt: datetime.datetime,
    mapping: dict[str, list[dict[str, Any]]],
) -> str | None:
  """Resolves the cluster machine for a GCE instance_id at event_dt.

  Args:
    inst_id: GCE instance ID string.
    event_dt: Timestamp of the system event.
    mapping: Mapping dictionary from instance_id to cluster machine rows.

  Returns:
    The resolved cluster machine identifier, or None if no valid mapping row
    exists for inst_id.
  """
  rows = mapping.get(inst_id, [])
  valid_rows = [row for row in rows if row.get("host_id")]
  if not valid_rows:
    return None
  if len(valid_rows) == 1:
    return str(valid_rows[0]["host_id"])
  best_machine = str(valid_rows[0]["host_id"])
  best_diff = None
  for row in valid_rows:
    ls_dt = _parse_iso_ts(str(row.get("last_seen", "")))
    if ls_dt is not None:
      diff = abs((event_dt - ls_dt).total_seconds())
      if best_diff is None or diff < best_diff:
        best_diff = diff
        best_machine = str(row["host_id"])
  return best_machine


def _max_in_rolling_24h(timestamps: list[datetime.datetime]) -> int:
  """Returns the maximum number of events within any rolling 24-hour window."""
  if not timestamps:
    return 0
  sorted_ts = sorted(timestamps)
  max_count = 0
  left = 0
  for right, dt_r in enumerate(sorted_ts):
    while (dt_r - sorted_ts[left]).total_seconds() > _ROLLING_WINDOW_SEC:
      left += 1
    max_count = max(max_count, right - left + 1)
  return max_count


def _recommend_next_step(
    strikes_14d: int,
    outages_24h: int,
    maint_24h: int,
    had_mass_event: bool,
) -> str:
  """Returns recommended remediation step from strike and outage counts."""
  if strikes_14d >= 3:
    return (
        "Escalate to Cloud TPU support for host replacement and hardware tray "
        "inspection, citing prior incidents, after collecting logs."
    )
  if strikes_14d == 2:
    return (
        "Customer project/cluster admin should file report-and-replace "
        "([[isolate-node-via-rr-label]]) with detailed template T2 notes in "
        "the referenced bug."
    )
  if strikes_14d <= 1 and outages_24h >= 3:
    if had_mass_event and strikes_14d == 0:
      return (
          "All outages occurred during mass events; track via the mass event "
          "ticket rather than per-host strikes."
      )
    return (
        "Customer project/cluster admin should keep the node out of slices "
        "([[taint-node]]) and record time, symptom, and chip."
    )
  if maint_24h >= 3:
    return (
        "Host has repeated upcomingMaintenance notices; keep large runs off it "
        "and trigger maintenance while idle or ask GCE maintenance oncall via "
        "Cloud TPU oncall to clear the notice."
    )
  return "Monitor host health and keep node tainted if further outages occur."


def detect_flapping_machines(
    events_file: str,
    mapping_file: str = "",
    limit_used: int = 50000,
) -> dict[str, Any]:
  """Analyzes GCE system_event JSON entries for flapping and mass events."""
  notes: list[str] = [
      "Thresholds (outages_24h >= 3, strikes_14d >= 2, maintenance_windows_24h"
      " >= 3) are heuristics from past TPU7x repairs, not SLOs."
  ]
  if not events_file or not os.path.isfile(events_file):
    return {
        "flapping_machines": [],
        "mass_events": [],
        "truncated": True,
        "notes": notes + [f"Events file not found: {events_file}"],
    }

  with open(events_file, "r", encoding="utf-8", errors="replace") as f:
    try:
      raw_events = json.load(f)
    except json.JSONDecodeError as exc:
      return {
          "flapping_machines": [],
          "mass_events": [],
          "truncated": True,
          "notes": notes + [f"Invalid JSON in events_file: {exc}"],
      }

  if not isinstance(raw_events, list):
    raw_events = []

  truncated = bool(limit_used <= 0 or len(raw_events) >= limit_used)
  if truncated:
    notes.append(
        f"Entry count ({len(raw_events)}) reached or exceeded limit_used"
        f" ({limit_used}); oldest events may be missing (truncation is not"
        " absence)."
    )

  mapping = _load_mapping(mapping_file, notes)
  unmapped_ids: set[str] = set()

  parsed_entries: list[dict[str, Any]] = []
  instanceless_entries: list[dict[str, Any]] = []

  for entry in raw_events:
    if not isinstance(entry, dict):
      continue
    ts_str = str(entry.get("timestamp", ""))
    dt = _parse_iso_ts(ts_str)
    if dt is None:
      continue
    proto = entry.get("protoPayload", {}) or {}
    method_name = str(proto.get("methodName", ""))
    res_labels = (entry.get("resource", {}) or {}).get("labels", {}) or {}
    inst_id = str(res_labels.get("instance_id", "")).strip()
    meta = proto.get("metadata", {}) or {}
    node_pool = str(meta.get("instance_group_manager_name", "")).strip()
    window_start = str(
        meta.get("windowStartTime", meta.get("maintenanceWindowStartTime", ""))
    ).strip()

    if not inst_id:
      instanceless_entries.append({
          "dt": dt,
          "timestamp": ts_str,
          "methodName": method_name,
      })
      continue

    host_id = _resolve_host_machine(inst_id, dt, mapping)
    if host_id is None:
      unmapped_ids.add(inst_id)
    key = host_id if host_id else f"instance:{inst_id}"
    parsed_entries.append({
        "dt": dt,
        "timestamp": ts_str,
        "methodName": method_name,
        "instance_id": inst_id,
        "host_id": host_id,
        "key": key,
        "node_pool": node_pool,
        "window_start": window_start,
    })

  if unmapped_ids:
    notes.append(
        "Fallback instance_id keying used for unmapped ids:"
        f" {sorted(unmapped_ids)} (instance-id keying undercounts outages and"
        " strikes)."
    )

  parsed_entries.sort(key=lambda x: x["dt"])
  if parsed_entries:
    earliest_dt = parsed_entries[0]["dt"]
    latest_dt = parsed_entries[-1]["dt"]
    span_days = (latest_dt - earliest_dt).total_seconds() / _ROLLING_WINDOW_SEC
    notes.append(
        f"Covered span: {parsed_entries[0]['timestamp']} to "
        f"{parsed_entries[-1]['timestamp']} ({span_days:.2f} days)."
    )
  else:
    latest_dt = datetime.datetime.now(datetime.timezone.utc)

  # Group per-machine outages (merged within 15 min = 900s) and maintenance
  # windows.
  machine_outages: dict[str, list[dict[str, Any]]] = {}
  machine_maint: dict[str, list[datetime.datetime]] = {}
  machine_maint_seen_windows: dict[str, set[str]] = {}
  machine_meta: dict[str, dict[str, Any]] = {}

  for entry in parsed_entries:
    key = entry["key"]
    info = machine_meta.setdefault(
        key,
        {
            "instance_ids": set(),
            "host_id": entry["host_id"],
            "first_seen": entry["timestamp"],
            "last_seen": entry["timestamp"],
            "evidence": [],
        },
    )
    info["instance_ids"].add(entry["instance_id"])
    info["last_seen"] = entry["timestamp"]

    method_name = entry["methodName"]
    if method_name.endswith("hostError") or method_name.endswith(
        "terminateOnHostMaintenance"
    ):
      info["evidence"].append(f"{entry['timestamp']} {method_name}")
      outages = machine_outages.setdefault(key, [])
      if (
          outages
          and (entry["dt"] - outages[-1]["end_dt"]).total_seconds()
          <= _OUTAGE_MERGE_WINDOW_SEC
      ):
        outages[-1]["end_dt"] = entry["dt"]
        outages[-1]["end_ts"] = entry["timestamp"]
        outages[-1]["methods"].add(method_name)
        if entry["node_pool"]:
          outages[-1]["node_pools"].add(entry["node_pool"])
      else:
        outages.append({
            "key": key,
            "host_id": entry["host_id"],
            "instance_id": entry["instance_id"],
            "start_dt": entry["dt"],
            "end_dt": entry["dt"],
            "start_ts": entry["timestamp"],
            "end_ts": entry["timestamp"],
            "methods": {method_name},
            "node_pools": {entry["node_pool"]} if entry["node_pool"] else set(),
            "in_mass_event": False,
        })
    elif method_name.endswith("upcomingMaintenance"):
      info["evidence"].append(f"{entry['timestamp']} {method_name}")
      win_key = entry["window_start"] or entry["timestamp"]
      seen_wins = machine_maint_seen_windows.setdefault(key, set())
      if win_key not in seen_wins:
        seen_wins.add(win_key)
        machine_maint.setdefault(key, []).append(entry["dt"])

  # Flatten all outages to detect mass events (>= 5 distinct machines within 15
  # min).
  all_outages: list[dict[str, Any]] = []
  for outs in machine_outages.values():
    all_outages.extend(outs)
  all_outages.sort(key=lambda x: x["start_dt"])

  mass_spans: list[tuple[datetime.datetime, datetime.datetime]] = []
  for i, out_i in enumerate(all_outages):
    window_start = out_i["start_dt"]
    window_end = window_start + datetime.timedelta(
        seconds=_OUTAGE_MERGE_WINDOW_SEC
    )
    machines_in_win = set()
    max_dt = out_i["end_dt"]
    for j in range(i, len(all_outages)):
      if all_outages[j]["start_dt"] <= window_end:
        machines_in_win.add(all_outages[j]["key"])
        if all_outages[j]["end_dt"] > max_dt:
          max_dt = all_outages[j]["end_dt"]
      else:
        break
    if len(machines_in_win) >= 5:
      if mass_spans and window_start <= mass_spans[-1][1] + datetime.timedelta(
          seconds=_OUTAGE_MERGE_WINDOW_SEC
      ):
        prev_s, prev_e = mass_spans[-1]
        mass_spans[-1] = (prev_s, max(prev_e, max_dt))
      else:
        mass_spans.append((window_start, max_dt))

  mass_events: list[dict[str, Any]] = []
  for span_s, span_e in mass_spans:
    span_machines: set[str] = set()
    span_methods: set[str] = set()
    span_pools: set[str] = set()
    start_str = span_s.isoformat()
    end_str = span_e.isoformat()
    for out in all_outages:
      if span_s <= out["start_dt"] <= span_e:
        out["in_mass_event"] = True
        span_machines.add(out["host_id"] or out["instance_id"])
        span_methods.update(out["methods"])
        span_pools.update(out["node_pools"])
    ctx = [
        f"{ie['timestamp']} {ie['methodName']}"
        for ie in instanceless_entries
        if span_s <= ie["dt"] <= span_e
    ]
    mass_events.append({
        "start": start_str,
        "end": end_str,
        "machine_count": len(span_machines),
        "machines": sorted(span_machines),
        "methods": sorted(span_methods),
        "node_pools": sorted(span_pools),
        "context": ctx,
        "recommended_next_step": (
            "Treat as a single mass event (no per-host strikes); escalate to"
            " Cloud TPU support to check for zone-wide maintenance or repair"
            " holds."
        ),
    })

  # Evaluate per-machine thresholds
  cutoff_14d = latest_dt - datetime.timedelta(days=14)
  flapping_machines: list[dict[str, Any]] = []

  for key, info in sorted(machine_meta.items()):
    outs = machine_outages.get(key, [])
    outage_dts = [outage["start_dt"] for outage in outs]
    outages_24h = _max_in_rolling_24h(outage_dts)

    # Non-mass-event outages in the 14d window, >= 2h (7200s) apart
    strikes_14d = 0
    last_strike_dt: datetime.datetime | None = None
    had_mass_event = any(outage["in_mass_event"] for outage in outs)
    for outage in outs:
      if outage["in_mass_event"] or outage["start_dt"] < cutoff_14d:
        continue
      if (
          last_strike_dt is None
          or (outage["start_dt"] - last_strike_dt).total_seconds()
          >= _STRIKE_COOLDOWN_SEC
      ):
        strikes_14d += 1
        last_strike_dt = outage["start_dt"]

    maint_dts = machine_maint.get(key, [])
    maint_24h = _max_in_rolling_24h(maint_dts)

    reasons: list[str] = []
    if outages_24h >= 3:
      reasons.append(f"outages_24h={outages_24h} (>=3)")
    if strikes_14d >= 2:
      reasons.append(f"strikes_14d={strikes_14d} (>=2)")
    if maint_24h >= 3:
      reasons.append(f"maintenance_windows_24h={maint_24h} (>=3)")

    if reasons:
      flapping_machines.append({
          "instance_id": ",".join(sorted(info["instance_ids"])),
          "host_id": info["host_id"],
          "outages_24h": outages_24h,
          "strikes_14d": strikes_14d,
          "maintenance_windows_24h": maint_24h,
          "first_seen": info["first_seen"],
          "last_seen": info["last_seen"],
          "reasons": reasons,
          "evidence": info["evidence"],
          "recommended_next_step": _recommend_next_step(
              strikes_14d, outages_24h, maint_24h, had_mass_event
          ),
      })

  return {
      "flapping_machines": flapping_machines,
      "mass_events": mass_events,
      "truncated": truncated,
      "notes": notes,
  }


def main() -> int:
  """CLI entrypoint for detect_flapping_machines."""
  parser = argparse.ArgumentParser(
      description=(
          "Detect flapping TPU machines and mass events from GCE system events."
      )
  )
  parser.add_argument(
      "--events_file",
      default="",
      help="Path to JSON array of GCE system_event entries.",
  )
  parser.add_argument(
      "--mapping_file",
      default="",
      help="Path to JSONL mapping GCE instance_id to host_id.",
  )
  parser.add_argument(
      "--limit_used",
      type=int,
      default=50000,
      help="Query limit used when generating events_file.",
  )
  args = parser.parse_args()

  result = detect_flapping_machines(
      events_file=args.events_file,
      mapping_file=args.mapping_file,
      limit_used=args.limit_used,
  )
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
