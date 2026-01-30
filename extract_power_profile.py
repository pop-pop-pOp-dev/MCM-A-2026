"""
Extract power estimates from Android power_profile.xml.

Converts mA entries to W using a nominal battery voltage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def find_items(root: ET.Element, name: str) -> list[float]:
    values: list[float] = []
    for item in root.findall("item"):
        if item.attrib.get("name") == name:
            try:
                values.append(float(item.text))
            except (TypeError, ValueError):
                pass
    return values


def find_array(root: ET.Element, name: str) -> list[float]:
    values: list[float] = []
    for array in root.findall("array"):
        if array.attrib.get("name") == name:
            for v in array.findall("value"):
                try:
                    values.append(float(v.text))
                except (TypeError, ValueError):
                    pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract power_profile values.")
    parser.add_argument(
        "--profile",
        type=str,
        default=str(Path("datasets") / "aosp_power_profile" / "power_profile.xml"),
        help="Path to power_profile.xml",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=3.85,
        help="Nominal battery voltage for mA->W conversion.",
    )
    args = parser.parse_args()

    tree = ET.parse(args.profile)
    root = tree.getroot()

    wifi_on_ma = max(find_items(root, "wifi.on") + [0.0])
    wifi_active_ma = max(find_items(root, "wifi.active") + [0.0])
    gps_on_ma = max(find_items(root, "gps.on") + [0.0])

    cpu_cluster0 = find_array(root, "cpu.active.cluster0")
    cpu_cluster1 = find_array(root, "cpu.active.cluster1")

    def to_w(ma: float) -> float:
        return ma * args.voltage / 1000.0

    p_wifi_idle = to_w(wifi_on_ma)
    p_wifi_active = to_w(wifi_active_ma)
    p_gps = to_w(gps_on_ma)

    p_little = to_w(max(cpu_cluster0) if cpu_cluster0 else 0.0)
    p_big = to_w(max(cpu_cluster1) if cpu_cluster1 else 0.0)

    print("Derived power estimates (W):")
    print(f"wifi_idle_power: {p_wifi_idle:.6f}")
    print(f"wifi_active_power: {p_wifi_active:.6f}")
    print(f"gps_on_power: {p_gps:.6f}")
    print(f"P_little_max: {p_little:.6f}")
    print(f"P_big_max: {p_big:.6f}")

    print("\nRaw mA values from power_profile.xml:")
    print(f"wifi.on: {wifi_on_ma:.3f} mA")
    print(f"wifi.active: {wifi_active_ma:.3f} mA")
    print(f"gps.on: {gps_on_ma:.3f} mA")
    if cpu_cluster0:
        print(f"cpu.active.cluster0 max: {max(cpu_cluster0):.3f} mA")
    if cpu_cluster1:
        print(f"cpu.active.cluster1 max: {max(cpu_cluster1):.3f} mA")

    print("\nYAML snippet:")
    print(f"wifi_idle_power: {p_wifi_idle:.6f}")
    print(f"wifi_active_power: {p_wifi_active:.6f}")
    print(f"gps_on_power: {p_gps:.6f}")
    print(f"P_little_max: {p_little:.6f}")
    print(f"P_big_max: {p_big:.6f}")


if __name__ == "__main__":
    main()
