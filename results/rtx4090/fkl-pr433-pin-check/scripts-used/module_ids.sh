#!/bin/bash
# The anonymous-namespace module ids (_GLOBAL__N__<8 hex>_<len>_<file>_<8 hex>) that nvcc put in the symbol names of each binary given (nm; the
# binaries are only read), with the binary's sha256 and the commits its build information carries.
#   module_ids.sh <binary> ...
for b in "$@"; do
  echo "$b"
  echo "  sha256 $(sha256sum "$b" | cut -d' ' -f1); commits $(strings "$b" | grep -E '^[0-9a-f]{40}(-dirty)?$|^unknown$' | tr '\n' ' ')"
  echo "  module ids: $(nm "$b" | grep -o -E '_GLOBAL__N__[0-9a-f]{8}_[0-9]+_[A-Za-z0-9_]+_[0-9a-f]{8}' | sort -u | tr '\n' ' ')"
done
