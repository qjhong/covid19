#!/bin/csh
set dir = proj


cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "$dir"/"$state"_proj | grep '2021 11 23'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -21
cp summary summary_Jul24

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "$dir"/"$state"_proj | grep '2021 11 28'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -21
cp summary summary_Jul29

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "$dir"/"$state"_proj | grep '2021 12 8'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -21
cp summary summary_Aug19

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "$dir"/"$state"_proj | grep '2022 3 3'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -21
cp summary summary_Feb28

