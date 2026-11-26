#!/usr/bin/env bash
# compare.sh <run folder A> <run folder B>: the outputs of two runs of dpps_before_after (A built against b10fc66, B against the #435 head),
# compared byte for byte. Both outputs.tsv must list the same (back-end, case) lines; each case's .bin must be the same bytes (cmp) and have the
# same sha256 and the same number of sentinel values in A and B, and each output must have been written (fewer sentinel values than values;
# the columns of outputs.tsv: back-end, case, bytes, sha256, sentinel values, values).
# Prints one line per case and a total; exit 0 = every case identical. No GPU; it only reads files.
set -u
A=${1:?usage: compare.sh <run A> <run B>}; B=${2:?usage: compare.sh <run A> <run B>}
ka=$(cut -f1,2 "$A/outputs.tsv" | sort); kb=$(cut -f1,2 "$B/outputs.tsv" | sort)
if [ "$ka" != "$kb" ]; then echo "the two runs list different cases:"; diff <(echo "$ka") <(echo "$kb"); exit 1; fi
same=0; differ=0
while IFS=$'\t' read -r backend name bytes sha left values; do
  lineB=$(awk -F'\t' -v b="$backend" -v n="$name" '$1 == b && $2 == n' "$B/outputs.tsv")
  bytesB=$(echo "$lineB" | cut -f3); shaB=$(echo "$lineB" | cut -f4); leftB=$(echo "$lineB" | cut -f5)
  file="${backend}_${name}.bin"
  if cmp -s "$A/$file" "$B/$file" && [ "$sha" = "$shaB" ] && [ "$bytes" = "$bytesB" ] && [ "$left" = "$leftB" ] && [ "$left" -lt "$values" ]; then
    verdict=identical; same=$((same + 1))
  else
    verdict=DIFFERENT; differ=$((differ + 1))
  fi
  echo -e "$verdict\t$backend\t$name\t$values values\t$bytes bytes\t$left sentinel values\t$sha"
done < "$A/outputs.tsv"
echo "TOTAL: $((same + differ)) outputs, $same identical, $differ different"
[ $differ = 0 ]
