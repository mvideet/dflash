#!/bin/bash
# Aggregate all v8/v7 log results into a single summary table.
cd "$(dirname "$0")"

LOGDIR="logs/v8"
echo "config                              speedup     tau      nodes"
echo "--------------------------------  ---------  -------  -------"
for LOG in ${LOGDIR}/*.log; do
    [ -f "${LOG}" ] || continue
    TAG=$(basename "${LOG}" .log)
    SPD=$(grep "Decoding speedup:" "${LOG}" | tail -1 | awk '{print $3}')
    TAU=$(grep "Average Acceptance length:" "${LOG}" | tail -1 | awk '{print $4}')
    NODES=$(grep "Average tree node count:" "${LOG}" | tail -1 | awk '{print $5}')
    printf "%-34s  %-8s  %-7s  %-7s\n" "${TAG}" "${SPD:-FAIL}" "${TAU:-FAIL}" "${NODES:-FAIL}"
done
