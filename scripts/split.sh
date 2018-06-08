#!/usr/bin/env bash

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
        NETDIR=$VALUE
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

echo "monitor start"

# clear test and train split dirs
rm -rf $TESTDIR $TRAINDIR
mkdir -vp $TESTDIR $TRAINDIR

n=0
let overhead="$WINSIZE / 10"
let max="$WINSIZE + $overhead"
overhead_train=$(echo "scale=1;($TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)
overhead_test=$(echo "scale=1;(1 - $TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)

inotifywait -q -m -e close_write $NETDIR | mbuffer -m 10M |
while read -r path event file
do
  if [[ $file = training.*.gz ]]
  then
    let "n++"

    id=$(echo $file | cut -d'.' -f 2)
    let x="$id % 100 + 1"

    if [ $x -gt $TRAINPCT ]
    then
      echo -n "*"
      target=$TESTDIR/$file
    else
      echo -n "T"
      target=$TRAINDIR/$file
    fi

    cp -a $path/$file $target

    if [ $n -eq $max ]
    then
      ls -rt $TESTDIR | head -n $overhead_test | xargs -I{} rm -f $TESTDIR/{}
      ls -rt $TRAINDIR | head -n $overhead_train | xargs -I{} rm -f $TRAINDIR/{}
      let "n -= $overhead"
      echo -n "-"
    fi
  fi
done

