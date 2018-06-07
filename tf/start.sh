#!/usr/bin/env bash

set -e

CONFIGDIR=$1
GAMES=$2

ROOT="/work/lc0"
NETDIR="$ROOT/networks/upload"
GAMEFILE="$HOME/.lc0.dat"
RAMDISK="/ramdisk"

if [ ! -f "$GAMEFILE" ]
then
  echo "File $GAMEFILE must contain a single number, exiting now!"
  exit 1
fi

game_num=$(cat $GAMEFILE)
game_num=$((game_num + GAMES))
file="training.${game_num}.gz"

echo "Starting with '$file' as last game in window"

train() {
  unbuffer ./train.py --cfg=$1 --output=$2 2>&1 > "$ROOT/logs/$(date +%Y%m%d-%H%M%S).log"
  gzip -9 $2
  mv -v $2.gz $NETDIR
}


while true
do
  if [ -f "$ROOT/data/$file" ]
  then
    echo ""

    # prepare ramdisk
    rm -rf $RAMDISK/*
    cp -r $ROOT/split/{train,test} $RAMDISK

    # train all networks
    for netarch in 64x6 128x10
    do
      cfg="$CONFIGDIR/$netarch.yaml"
      echo "Training $netarch:"
      echo $(cat $cfg)
      train "$CONFIGDIR/$netarch.yaml" "${netarch}-$(date +"%Y_%m%d_%H%M_%S_%3N").txt"
    done

    # wait for next cycle
    echo $game_num > $GAMEFILE
    game_num=$((game_num + GAMES))
    file="training.${game_num}.gz"
    echo "Waiting for '$file'"
  else
    sleep 60
    echo -n "."
  fi
done
