#!/bin/csh
@ date = `date | awk '{print $3}'`
@ month = 6
@ enddate = 15
@ endmonth = 7
while ( $month < $endmonth || $date <= $enddate)
  set sum = `grep "2020 $month $date " *_proj | awk '{s+=$4} END {print s}'`
  echo 2020 $month $date $sum
  @ date = $date + 1
  if ( $date == 31 ) then
    @ date = 1
    @ month = $month + 1
  endif
end
