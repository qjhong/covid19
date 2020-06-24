#!/bin/csh
@ date = `date | awk '{print $3}'`
@ month = 6
@ enddate = 15
@ endmonth = 7
while ( $month < $endmonth || $date <= $enddate)
  set sum = `grep "2020 $month $date " *_proj | awk '{s+=$4} END {print s}'`
  set sum_l = `grep "2020 $month $date " *_proj | awk '{s+=$5} END {print s}'`
  set sum_h = `grep "2020 $month $date " *_proj | awk '{s+=$6} END {print s}'`
  echo 2020 $month $date $sum $sum_l $sum_h
  @ date = $date + 1
  if ( $date == 31 ) then
    @ date = 1
    @ month = $month + 1
  endif
end
