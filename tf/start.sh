#!/usr/bin/env bash

set -e

NETARCHS=(64x6)
REPO="origin"
ROOT="/work/lc0"
NETDIR="$ROOT/networks/upload"
GAMEFILE="$HOME/.lc0.dat"
LATESTFILE="$HOME/.lc0.latest.dat"
RAMDISK="/ramdisk"
MIN_GAP=10

function usage()
{
  echo "Starts a training pipeline"
  echo ""
  echo "./start.sh"
  echo "  -h --help"
  echo "  -c --cfg    The configuration directory"
  echo "  -g --games  The number of games between training cycles"
  echo "  -b --branch The git branch to push configs to"
  echo ""
  echo "Example: ./start.sh -c=/tmp/cfgdir -g=40000 -b=test"
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
    -c | --cfg)
      CONFIGDIR=$VALUE
      ;;
    -g | --games)
      GAMES=$VALUE
      ;;
    *)
      echo "ERROR: unknown parameter \"$PARAM\""
      usage
      exit 1
      ;;
  esac
  shift
done
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
  mv -v $2.pb.gz $NETDIR
}

delay_count=$((MIN_GAP+1))

while true
do
  latest_num=0
  if [ -f "$LATESTFILE" ]
  then
    latest_num=$(cat $LATESTFILE)
  fi
  if [[ $delay_count -gt $MIN_GAP && ( $latest_num -gt $game_num || -f "$ROOT/data-rescored/$file" ) ]]
  then
    if [ $latest_num -gt $game_num ]
    then
      game_num=$latest_num
    fi
    echo ""

    # prepare ramdisk
    (
    flock -e 200
    rsync -aq --delete-during $ROOT/split/{train,test} $RAMDISK
    ) 200>$LC0LOCKFILE

    # train all networks
    for netarch in ${NETARCHS[@]}
    do
      echo "Training $netarch:"
      train "$CONFIGDIR/$netarch.yaml" "${netarch}-$(date +"%Y_%m%d_%H%M_%S_%3N")"
    done

    # wait for next cycle
    echo $game_num > $GAMEFILE
    game_num=$((game_num + GAMES))
    file="training.${game_num}.gz"
    echo "Waiting for '$file'"
    delay_count=1
  else
    echo -n "."
    sleep 60
    delay_count=$((delay_count+1))
  fi
done
