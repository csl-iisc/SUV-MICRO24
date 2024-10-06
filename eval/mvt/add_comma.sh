#/bin/bash

file=$1
echo $file
sed -i '$!s/$/,/' $file
