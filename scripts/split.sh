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

let n="$(ls $TESTDIR | wc -l) + $(ls $TRAINDIR | wc -l)"
let overhead="$WINSIZE / 10"
let max="$WINSIZE + $overhead + 100"
overhead_train=$(echo "scale=1;($TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)
overhead_test=$(echo "scale=1;(1 - $TRAINPCT / 100) * $overhead" | bc | cut -d'.' -f1)

echo ""
echo "start splitter, found $n games"
echo "  max chunks: $max"
echo "  max test:   $overhead_test"
echo "  max train:  $overhead_train"
echo ""


process() {
  local dir=$1
  local file=$2

  if [[ $file = training.*.gz ]]
  then
    # check if the file already exists in the desired location
    if [ -f "$TESTDIR/$file" ] || [ -f "$TRAINDIR/$file" ]
    then
      echo -n "."
      return
    fi

    # compute basic file integrity check
    size=$(zcat $dir/$file | wc -c)
    let rem="size % 8276"
    
    if [[ $size -eq 0 ]] || [[ $rem -ne 0 ]]
    then
      echo -n "X"
      return
    fi

    # new correct file, put in correct directory
    let "n++"

    id=$(echo $file | cut -d'.' -f 2)
    let x="$id % 100 + 1"

    if [ $x -gt $TRAINPCT ]
    then
      target=$TESTDIR/$file
      echo -n "T"
    else
      target=$TRAINDIR/$file
      echo -n "*"
    fi

    ln $dir/$file $target

    # exceeding max buffer size, remove overhead
    if [ $n -gt $max ]
    then
      (
      flock -e 200
      ls -rt $TESTDIR | head -n $overhead_test | xargs -I{} rm -f $TESTDIR/{}
      ls -rt $TRAINDIR | head -n $overhead_train | xargs -I{} rm -f $TRAINDIR/{}
      ) 200>$LC0LOCKFILE
      let "n -= $overhead"
    fi
  fi
}


echo -n "processing data in '$INPUTDIR'..."
for f in $(ls -1rt $INPUTDIR | tail -n $WINSIZE)
do
  file=$(basename $f)
  process $INPUTDIR $file
done
echo "[done]"

echo "monitoring '$INPUTDIR'"
inotifywait -q -m -e moved_to -e close_write $INPUTDIR | mbuffer -m 10M |
  while read dir event file
  do
    if [[ $event = MOVED_TO ]] || [[ $event = CLOSE_WRITE ]]
    then
      process $INPUTDIR $file
    fi
  done
