#!/bin/csh
set dir = proj


cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "$dir"/"$state"_proj | grep '2021 7 24'`
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
  set line = `cat "$dir"/"$state"_proj | grep '2021 7 29'`
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
  set line = `cat "$dir"/"$state"_proj | grep '2021 8 19'`
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
  set line = `cat "$dir"/"$state"_proj | grep '2021 2 28'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -21
cp summary summary_Feb28

