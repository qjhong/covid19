#!/bin/csh

rm *_proj
cp list list_tmp
@ l = `cat list_tmp | wc -l`
while ( $l > 0 )
  set state = `head -1 list_tmp`
  python covid19_state.py $state
  sed '1d' list_tmp > list_tmp2
  cp list_tmp2 list_tmp
  sed "s/NY/$state/g" html.module > html_tmp
  mv html_tmp $state.html
  @ l = `cat list_tmp | wc -l`
end

mv *.png fig
mv *.html html
mv *_proj proj
rm tmp*

./summary.csh
