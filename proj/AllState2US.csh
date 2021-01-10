#!/bin/csh
@ date = `date | awk '{print $3}'`
@ date = 28
@ month = 12
@ enddate = 28
@ endmonth = 2
@ year = 2020
@ endyear = 2021
while ( $year < $endyear || $month < $endmonth || $date <= $enddate )
  set sum = `grep "$year $month $date " *_proj | awk '{s+=$4} END {print s}'`
  set sum_l = `grep "$year $month $date " *_proj | awk '{s+=($4-$5)*($4-$5)/1000000} END {print s}'`
  set sum_l = `echo "sqrt($sum_l)*1000" | bc`
#  set sum_l = `echo "sqrt($sum_l)*100*sqrt(3.0)" | bc`
#  set sum_l = `grep "$year $month $date " *_proj | awk '{s+=$5} END {print s}'`
  set sum_h = `grep "$year $month $date " *_proj | awk '{s+=($6-$4)*($6-$4)/1000000} END {print s}'`
  set sum_h = `echo "sqrt($sum_h)*1000" | bc`
#  set sum_h = `echo "sqrt($sum_h)*100*sqrt(3.0)" | bc`
#  set sum_h = `grep "$year $month $date " *_proj | awk '{s+=$6} END {print s}'`
  echo $year $month $date $sum $sum_l $sum_h
  @ date = $date + 1
  if ( $date == 32 ) then
    @ date = 1
    @ month = $month + 1
    if $month == 13 then
      @ year = $year + 1
      @ month = 1
      @ day = 1
    endif
  endif
end
