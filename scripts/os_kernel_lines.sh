# Source this file. os_kernel_lines prints the two comment lines that open every window log, sampler file and phases file of a run:
#   # os: Ubuntu 26.04.1 LTS
#   # kernel: Linux 7.0.0-38-generic
# read at run time from /etc/os-release and uname (common/system_info.h writes the same two lines into the header of every CSV). Readers skip the lines that start with '#'.
os_kernel_lines() {
    local name
    name=$(. /etc/os-release 2> /dev/null && echo "${PRETTY_NAME:-unavailable}") || name=unavailable
    echo "# os: ${name:-unavailable}"
    echo "# kernel: $(uname -sr)"
}
