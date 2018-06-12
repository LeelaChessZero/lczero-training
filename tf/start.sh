#!/usr/bin/env bash

set -e

CONFIGDIR=$1
GAMES=$2

NETARCHS=(64x6 128x10 192x15)
ROOT="/work/lc0"
NETDIR="$ROOT/networks/upload"
GAMEFILE="$HOME/.lc0.dat"
RAMDISK="/ramdisk"

if [ ! -f "$GAMEFILE" ]
then
  echo "File $GAMEFILE must contain a single number, exiting now!"
  exit 1
fi

if [ -z "$LC0LOCKFILE" ]
then
  echo "env var LC0LOCKFILE not set"
  exit 1
fi

game_num=$(cat $GAMEFILE)
game_num=$((game_num + GAMES))
file="training.${game_num}.gz"

echo "Starting with '$file' as last game in window"

train() {
  unbuffer ./train.py --cfg=$1 --output=$2 2>&1 | tee "$ROOT/logs/$(date +%Y%m%d-%H%M%S).log"
  gzip -9 $2
  mv -v $2.gz $NETDIR
}


while true
do
  if [ -f "$ROOT/data/$file" ]
  then
    echo ""

    # prepare ramdisk
    (
    flock -e 200
    rsync -aq --delete-during $ROOT/split/{train,test} $RAMDISK
    ) 200>$LC0LOCKFILE

    # train all networks
    for netarch in ${NETARCHS[@]}
    do
      echo "Saving .yaml changes:"
      pushd $CONFIGDIR
      git add $netarch.yaml
      git commit -m "Configuration change"
      git push https://github.com/LeelaChessZero/lczero-training-conf.git &
      popd
      echo "Training $netarch:"
      train "$CONFIGDIR/$netarch.yaml" "${netarch}-$(date +"%Y_%m%d_%H%M_%S_%3N").txt"
    done

    # wait for next cycle
    echo $game_num > $GAMEFILE
    game_num=$((game_num + GAMES))
    file="training.${game_num}.gz"
    echo "Waiting for '$file'"
  else
    echo -n "."
    sleep 60
  fi
done
