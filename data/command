#!/bin/csh

cat $1_ED | cut -d\" -f14 | sed 's/L/ /g' | sed 's/M//' > $1_ED_tmp
mv $1_ED_tmp $1_ED
