#!/bin/csh
@ date = `date | awk '{print $3}'`
@ date = 24
@ month = 5
@ enddate = 1
@ endmonth = 7
while ( $month < $endmonth || $date <= $enddate)
  set sum = `grep "2020 $month $date " *_proj | awk '{s+=$4} END {print s}'`
  set sum_l = `grep "2020 $month $date " *_proj | awk '{s+=($4-$5)*($4-$5)/10000} END {print s}'`
  set sum_l = `echo "sqrt($sum_l)*100" | bc`
#  set sum_l = `echo "sqrt($sum_l)*100*sqrt(5)" | bc`
#  set sum_l = `grep "2020 $month $date " *_proj | awk '{s+=$5} END {print s}'`
  set sum_h = `grep "2020 $month $date " *_proj | awk '{s+=($6-$4)*($6-$4)/10000} END {print s}'`
  set sum_h = `echo "sqrt($sum_h)*100" | bc`
#  set sum_h = `echo "sqrt($sum_h)*100*sqrt(5)" | bc`
#  set sum_h = `grep "2020 $month $date " *_proj | awk '{s+=$6} END {print s}'`
  echo 2020 $month $date $sum $sum_l $sum_h
  @ date = $date + 1
  if ( $date == 31 ) then
    @ date = 1
    @ month = $month + 1
  endif
end
