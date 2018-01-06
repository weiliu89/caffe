#!/bin/bash

root_dir=$HOME/data/VOCdevkit/
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  echo "[test-cmd] : test \$dst_file '$dst_file' existing"
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    echo "[test-cmd] : editing image file $img_file"
    cp $dataset_file $img_file
    sed -i'.original' "s/^/$name\/JPEGImages\//g" $img_file
    sed -i'.original' "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i'.original' "s/^/$name\/Annotations\//g" $label_file
    sed -i'.original' "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    echo "[test-cmd] : gen image name and size"
    echo "[test-cmd] : exec : $bash_dir/../../build/tools/get_image_size"
    echo "[test-cmd] : \$root_dir : $root_dir"
    echo "[test-cmd] : \$dst_file : $dst_file"
    echo "[test-cmd] : \$bash_dir/\$dataset : $bash_dir/$dataset"
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    echo "[test-cmd] : shuffle trainval file"
    echo "[test-cmd] : target : $dst_file"
    rand_file=$dst_file.random
    echo "[test-cmd] : rand file : $rand_file"
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
