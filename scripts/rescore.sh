#!/usr/bin/env bash

set -e

ROOT="/work/lc0/dev2"
RESCORER="$HOME/bin/rescorer"

function usage()
{
  echo "Rescores stuff"
  echo ""
  echo "./rescore.sh"
  echo "  -h --help"
  echo ""
  echo "Example: ./rescore.sh"
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
    *)
      echo "ERROR: unknown parameter \"$PARAM\""
      usage
      exit 1
      ;;
  esac
  shift
done

rescore() {
  unbuffer $RESCORER rescore --threads=4 --syzygy-paths=/work/lc0/syzygy/:/wdl/syzygy/wdl/:/wdl/syzygy/dtz/ --input="$ROOT/data-staged" --output="$ROOT/data-rescored" 2>&1 | tee "$ROOT/rescore-logs/$(date +%Y%m%d-%H%M%S).log"
}

while true
do
  rescore
  echo -n "."
  sleep 10
done
