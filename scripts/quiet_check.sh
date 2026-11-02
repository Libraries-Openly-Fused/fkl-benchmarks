# Source this file. busy_processes prints, one `ps` line each, the processes that must not be running during a quiet measurement window: compilers and profilers, the benchmark programs, and the window and runner scripts.
# It matches the executable only (the process name, the path of argv[0], or the script that bash or python runs), never a word somewhere in a command line, so a watcher, a grep or a message cannot match by accident.
# The process that sources this file (the window script) and its children are not listed: `$(busy_processes)` forks a subshell that has the command line of the script. Usage in a window script:
#   source $FB/scripts/quiet_check.sh; busy=$(busy_processes); if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
busy_processes() {
    ps -eo pid=,ppid=,comm=,args= | awk -v self=$$ '
        $1 == self || $2 == self { next }
        $3 ~ /^(nvcc|cicc|ptxas|cc1plus|cudafe\+\+|fatbinary|nvlink|nsys|ncu|ncu-ui|nsys-ui)$/ { print; next }
        $4 ~ /\/bin\/[^\/]+$/ && ($4 ~ /\/build[^\/]*\/(.*\/)?bin\// || $4 ~ /\/e0-frozen-[0-9a-f]+\/bin\//) { print; next }
        $3 ~ /^(bash|sh)$/ && $5 ~ /(^|\/)(gpu-window-[^\/]*|run_e0|pin_transfer)\.sh$/ { print; next }
        $3 ~ /^python/ && ($5 ~ /\/(pytorch-attention|triton)\/[^\/]*\.py$/ || $6 ~ /\/(pytorch-attention|triton)\/[^\/]*\.py$/) { print; next }
    '
}
