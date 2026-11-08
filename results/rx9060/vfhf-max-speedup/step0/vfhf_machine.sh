#!/usr/bin/env bash
# vfhf_machine.sh: the machine facts of step 0, read only (sysfs, /proc): the runner, the load, the GPU's clocks, links and use, ASPM, the environment
set -u
shopt -s nullglob
echo "== machine facts, $(date -u +%FT%TZ)"
echo "runner service: $(systemctl is-active actions.runner.Libraries-Openly-Fused.asusb550mlinux.service 2>/dev/null)"
echo "Runner.Worker processes: $(pgrep -xc Runner.Worker)"
echo "Runner.Listener processes: $(pgrep -xc Runner.Listener)"
echo "loadavg: $(cut -d' ' -f1-3 /proc/loadavg) (the idle rule of the runs: no Runner.Worker and a 1-minute load below 0.8)"
for card in /sys/class/drm/card*/device; do
  [ "$(cat $card/vendor 2>/dev/null)" = "0x1002" ] || continue
  dev=$(readlink -f $card)
  echo "AMD GPU: $(basename $dev); gpu_busy_percent $(cat $card/gpu_busy_percent 2>/dev/null); power_dpm_force_performance_level $(cat $card/power_dpm_force_performance_level 2>/dev/null)"
  for hw in $card/hwmon/hwmon*; do
    echo "  sclk $(($(cat $hw/freq1_input) / 1000000)) MHz, mclk $(($(cat $hw/freq2_input) / 1000000)) MHz, edge $(($(cat $hw/temp1_input) / 1000)) C"
  done
  echo "  card link (the GPU function's own): $(cat $card/current_link_speed) x$(cat $card/current_link_width)"
  rp=$(echo $dev | sed -E 's#^(/sys/devices/pci[0-9a-f:]+/[0-9a-f:.]+)/.*#\1#')
  echo "  host link (root port $(basename $rp)): $(cat $rp/current_link_speed) x$(cat $rp/current_link_width)"
done
echo "processes of this user holding /dev/kfd: $(for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q '/dev/kfd' && echo -n "$(basename $p) "; done)"
echo "/sys/module/pcie_aspm/parameters/policy: $(cat /sys/module/pcie_aspm/parameters/policy)"
for d in 0000:00:03.1 0000:0b:00.0 0000:0c:00.0 0000:0d:00.0; do
  l=/sys/bus/pci/devices/$d/link
  if [ -d "$l" ]; then
    vals=""; for f in "$l"/*; do vals="$vals$(basename "$f")=$(cat "$f" 2>/dev/null) "; done
    echo "$d link/: ${vals:-present, empty}"
  else
    echo "$d link/: absent"
  fi
done
echo "HIP_/HSA_/ROC*/AMD_ variables in the environment the run inherits: $(env | grep -cE '^(HIP_|HSA_|ROC|AMD_)')"
echo "VFHF_ variables in that environment: $(env | grep -cE '^VFHF_')"
