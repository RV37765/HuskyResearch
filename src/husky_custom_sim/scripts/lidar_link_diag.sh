#!/bin/bash
# lidar_link_diag.sh -- read-only look at the VLP-16's periodic link drops.
#
# 2026-11-03: the LiDAR's port (enp1s0, bridged into br0) dropped its link in
# bursts of three (+0, +10, +31 s) every ~13 min 38 s since boot, parked or
# driving, and each burst cost ~39 s of /velodyne_points. This prints what is
# needed to tell a sensor restart from the robot resetting its own port.
# Changes nothing.
#
#   bash lidar_link_diag.sh            (if the journal part is empty, run it with sudo)

IF=${1:-enp1s0}
echo "== boot time: $(uptime -s)   now: $(date '+%F %T')"

echo "== link-down events on $IF since boot"
dmesg -T | grep "$IF NIC Link is Down" | sed 's/^\[\(.*\)\].*/\1/'

echo "== burst spacing, and when the next burst is due"
boot=$(date -d "$(uptime -s)" +%s)
dmesg | grep "$IF NIC Link is Down" | sed 's/^\[ *\([0-9.]*\)\].*/\1/' | awk -v boot="$boot" '
  { t = $1; if (n == 0 || t - last > 120) starts[++n] = t; last = t }
  END {
    for (i = 2; i <= n; i++) printf "  burst %d starts %.0f s after the previous one\n", i, starts[i] - starts[i-1]
    # burst 1 is usually the link coming up at boot, so the period uses bursts 2..n
    if (n >= 3) period = (starts[n] - starts[2]) / (n - 2); else if (n == 2) period = starts[2] - starts[1]
    if (n >= 2) printf "NEXT %d %.0f\n", boot + starts[n] + period, period
  }' | while IFS= read -r line; do
    case "$line" in
      NEXT*) set -- $line; echo "  mean period $3 s -> next burst due about $(date -d @$2 +%T)" ;;
      *) echo "$line" ;;
    esac
  done

echo "== who is on the robot network (a Velodyne MAC starts 60:76:88)"
ip neigh show dev br0 2>/dev/null
echo "== bridge members and link state"
bridge link 2>/dev/null || ip -br link show

echo "== NIC settings ($IF)"
ethtool $IF 2>/dev/null | grep -E "Speed|Duplex|Auto-negotiation|Link detected"
ethtool --show-eee $IF 2>/dev/null | grep -iE "EEE status|active"

echo "== does a network manager own $IF?"
command -v nmcli >/dev/null && nmcli -t device status 2>/dev/null | grep -E "$IF|br0"
networkctl list 2>/dev/null | grep -E "$IF|br0"
ls /etc/netplan/ 2>/dev/null && grep -vE "^\s*#" /etc/netplan/*.yaml 2>/dev/null | head -40

echo "== system journal around the last burst (anything besides the kernel driver?)"
last=$(dmesg -T | grep "$IF NIC Link is Down" | tail -3 | head -1 | sed 's/^\[\(.*\)\].*/\1/')
if [ -n "$last" ]; then
  e=$(date -d "$last" +%s)
  journalctl --since "$(date -d @$((e - 20)) '+%F %T')" --until "$(date -d @$((e + 60)) '+%F %T')" --no-pager 2>/dev/null \
    | grep -vE "row_follower" | tail -40
fi
