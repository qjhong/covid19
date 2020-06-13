#!/bin/csh

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat proj/"$state"_proj | grep '2020 6 13'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -13
cp summary summary_Jun10

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "proj/$state"_proj | grep '2020 6 20'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -11
cp summary summary_Jun20

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "proj/$state"_proj | grep '2020 6 30'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -11
cp summary summary_Jun30

cp list list_tmp
@ l = `cat list_tmp | wc -l`
rm summary
while ( $l > 0 )
  set state = `head -1 list_tmp`
  set line = `cat "proj/$state"_proj | grep '2020 7 15'`
  set number = `echo $line | awk '{print $4}'`
  echo $number $state >> summary
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  @ l = `cat list_tmp | wc -l`
end
cat summary | sort -n | tail -11
cp summary summary_Jul15
