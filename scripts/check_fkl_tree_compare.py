#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two runs of scripts/check_fkl_tree.sh (a baseline and a tree under test) group by group.

  python scripts/check_fkl_tree_compare.py BASELINE_DIR[,BASELINE_DIR2...] CANDIDATE_DIR[,CANDIDATE_DIR2...]

Each argument is one or more output folders of check_fkl_tree.sh (comma separated: the groups of a run can be spread over several folders); every <group>-report.csv found is read. Prints, per group: the number
of targets in both, the REGRESSIONS (OK in the baseline, FAIL or NOT_BUILT in the candidate, with the first error line of the candidate), the targets that failed in both, the targets only in the candidate (new) and
only in the baseline (removed). REJECTED_OK (an expected rejection that held, see check_fkl_tree.sh) counts as OK. Exit status 1 when there is a regression.
"""
import csv
import glob
import os
import sys

PASSING = ("OK", "REJECTED_OK")


def load(dirs):
    out = {}
    for d in dirs.split(","):
        for f in sorted(glob.glob(os.path.join(d, "*-report.csv"))):
            for r in csv.DictReader(open(f)):
                out.setdefault(r["group"], {})[r["target"]] = r
    return out


def main():
    base, cand = load(sys.argv[1]), load(sys.argv[2])
    regress = 0
    for g in sorted(set(base) | set(cand)):
        b, c = base.get(g, {}), cand.get(g, {})
        both = sorted(set(b) & set(c))
        reg = [t for t in both if b[t]["status"] in PASSING and c[t]["status"] not in PASSING]
        bothfail = [t for t in both if b[t]["status"] not in PASSING and c[t]["status"] not in PASSING]
        fixed = [t for t in both if b[t]["status"] not in PASSING and c[t]["status"] in PASSING]
        new, removed = sorted(set(c) - set(b)), sorted(set(b) - set(c))
        nok = sum(1 for t in c if c[t]["status"] in PASSING)
        print(f"== {g}: baseline {len(b)} targets ({sum(1 for t in b if b[t]['status'] in PASSING)} OK), candidate {len(c)} targets ({nok} OK); in both {len(both)}")
        for t in reg:
            regress += 1
            print(f"  REGRESSION {t}: {c[t]['status']}: {c[t]['first_error']}")
        for t in bothfail:
            print(f"  fails in both {t}: {c[t]['first_error']}")
        for t in fixed:
            print(f"  fixed {t}")
        print(f"  only in the candidate ({len(new)}): {', '.join(new)}" + "".join(f"\n    {t}: {c[t]['status']} {c[t]['first_error']}" for t in new if c[t]['status'] not in PASSING))
        print(f"  only in the baseline ({len(removed)}): {', '.join(removed)}")
    print(f"regressions: {regress}")
    return 1 if regress else 0


if __name__ == "__main__":
    sys.exit(main())
