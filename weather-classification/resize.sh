#!/usr/bin/expect -f

src_dir=$(realpath -s $1)
dst_dir=$(realpath -s $2)

#  Create directories
for weather in "Fog" "Night" "Rain" "Snow"; do
  mkdir $dst_dir/$weather/ > /dev/null 2>&1
  for task in "Test" "Train" "Validation"; do
    mkdir $dst_dir/$weather/${weather}_${task}/ > /dev/null 2>&1
  done
done

#  Resize images to 224 x 224
for weather in "Fog" "Night" "Rain" "Snow"; do
  for task in "Test" "Train" "Validation"; do
    for filepath in $src_dir/$weather/RGB/${weather}_${task}/*;do
      convert $src_dir/$weather/RGB/${weather}_${task}/`basename "$filepath"` -resize 224x224! $dst_dir/$weather/${weather}_${task}/`basename "$filepath"`
    done
  done
done