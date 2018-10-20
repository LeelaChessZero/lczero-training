#!/usr/bin/env bash

RECORDSIZE=8276 # size in bytes of a record (s, pi, v)

function usage()
{
  echo "Watches a directory and copies data to train/test set"
  echo ""
  echo "./split.sh"
  echo "  -h --help"
  echo "  -i --input   The directory where chunks arrive"
  echo "  -o --output  The output directory"
  echo "  -n --window  window size of test + train"
  echo "  -t --train   The training percentage in {1,...,100}"
  echo ""
  echo "Example: ./split.sh -i=/tmp -o=/out -n=2000 -t=95"
  echo ""
}

while [ "$1" != "" ]
do
  PARAM=`echo $1 | awk -F= '{print $1}'`
  VALUE=`echo $1 | awk -F= '{print $2}'`
  case $PARAM in
    -h | --help)
      usage
      exit
      ;;
    -i | --input)
      INPUTDIR=$VALUE
      ;;
    -o | --output)
      TESTDIR="$VALUE/test"
      TRAINDIR="$VALUE/train"
      ;;
    -n | --window)
      WINSIZE=$VALUE
      ;;
    -t | --train)
      TRAINPCT=$VALUE
      ;;
    *)
      echo "ERROR: unknown parameter \"$PARAM\""
      usage
      exit 1
      ;;
  esac
  shift
done

if [ -z "$LC0LOCKFILE" ]
then
  echo "env var LC0LOCKFILE not set"
  exit 1
fi

# clear test and train split dirs
if [ ! -d "$TESTDIR" ] || [ ! -d "$TRAINDIR" ]
then
  rm -rf "$TESTDIR" "$TRAINDIR"
  mkdir -vp "$TESTDIR" "$TRAINDIR"
fi

let n_test="$(ls $TESTDIR | wc -l)"
let n_train="$(ls $TRAINDIR | wc -l)"
let n="$n_test + $n_train"
let overhead="$WINSIZE / 10"
let max="$WINSIZE + $overhead + 200"
max_train=$(echo "scale=1;($TRAINPCT / 100) * $max" | bc | cut -d'.' -f1)
max_test=$(echo "scale=1;(1 - $TRAINPCT / 100) * $max" | bc | cut -d'.' -f1)
overhead_train=$(echo "scale=1;($TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)
overhead_test=$(echo "scale=1;(1 - $TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)

echo ""
echo "start splitter, found $n games, $n_test test, $n_train train"
echo "  max chunks: $max"
echo "  max test:   $max_test, trim_by: $overhead_test"
echo "  max train:  $max_train, trim_by: $overhead_train"
echo ""


process() {
  local dir=$1
  local file=$2

  if [[ $file = training.*.gz ]]
  then
    # compute basic file integrity check
    size=$(zcat $dir/$file | wc -c)
    let rem="size % $RECORDSIZE"
    
    if [[ $size -eq 0 ]] || [[ $rem -ne 0 ]]
    then
      echo -n "X"
      return
    fi

    # new file, put hard link in correct directory
    let "n++"

    id=$(echo $file | cut -d'.' -f 2)
    let hash_index="$id % 100 + 1"

    if [ $hash_index -gt $TRAINPCT ]
    then
      let "n_test++"
      target=$TESTDIR/$file
      echo -n "T"
    else
      let "n_train++"
      target=$TRAINDIR/$file
      echo -n "*"
    fi

    ln $dir/$file $target

    # exceeding max buffer size for either, lock and remove overhead as appropriate
    if [ $n_test -gt $max_test ] || [ $n_train -gt $max_train ]
    then
      (
      flock -e 200
      if [ $n_test -gt $max_test ]
      then
        ls -rt $TESTDIR | head -n $overhead_test | xargs -I{} rm -f $TESTDIR/{}
        echo -n "-"
      fi
      if [ $n_train -gt $max_train ]
      then
        ls -rt $TRAINDIR | head -n $overhead_train | xargs -I{} rm -f $TRAINDIR/{}
        echo -n "_"
      fi
      ) 200>$LC0LOCKFILE
      if [ $n_test -gt $max_test ]
      then
        let "n -= $overhead_test"
        let "n_test -= $overhead_test"
        echo -n "-"
      fi
      if [ $n_train -gt $max_train ]
      then
        let "n -= $overhead_train"
        let "n_train -= $overhead_train"
        echo -n "_"
      fi

    fi
  fi
}


echo "processing '$INPUTDIR'"
for file in $(./diff.py -i $INPUTDIR -w $WINSIZE $TRAINDIR $TESTDIR)
do
  process $INPUTDIR $file
done

echo -e "\nmonitoring '$INPUTDIR'"
inotifywait -q -m -e moved_to -e close_write $INPUTDIR | mbuffer -m 10M |
  while read dir event file
  do
    if [ -f "$TESTDIR/$file" ] || [ -f "$TRAINDIR/$file" ]
    then
      continue
    fi

    process $INPUTDIR $file
  done
