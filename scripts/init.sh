#!/usr/bin/env bash

set -e

WINDOWSIZE=80000
ROOT="/work/lc0"

echo "Cleaning up data directory"
rm -rf $ROOT/data
mkdir -v $ROOT/data

echo "Hard link $WINDOWSIZE seed files in $ROOT/data"
i=1
for file in $(find $ROOT/seed/data-* -name '*.gz' | shuf)
do
  ln $file $ROOT/data/training.$i.gz
  if [[ $i = $WINDOWSIZE ]]
  then
    break
  fi
  let i="i + 1"
done

echo "Set $HOME/.lc0.dat to $WINDOWSIZE"
echo $WINDOWSIZE > $HOME/.lc0.dat

rm -rf $ROOT/split
mkdir -vp $ROOT/split/{test,train}
let testsize="$WINDOWSIZE / 10"
let trainsize="$WINDOWSIZE - $testsize"
echo "Create $ROOT/split/test ($testsize) and $ROOT/split/train ($trainsize)"
ls -1 -U $ROOT/data | head -n $trainsize | xargs -I{} ln $ROOT/data/{} $ROOT/split/train/{}
ls -1 -U $ROOT/data | tail -n $testsize | xargs -I{} ln $ROOT/data/{} $ROOT/split/test/{}
