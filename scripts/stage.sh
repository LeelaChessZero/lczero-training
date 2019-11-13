#!/usr/bin/env bash

set -e

function usage()
{
  echo "Moves arriving data to a directory so rescorer can assume all files are complete"
  echo ""
  echo "./stage.sh"
  echo "  -h --help"
  echo "  -i --input   The monitoring directory"
  echo "  -o --output   The directory where output should go"
  echo ""
  echo "Example: ./stage.sh -i data -o data-staged"
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
      OUTPUTDIR=$VALUE
      ;;
    *)
      echo "ERROR: unknown parameter \"$PARAM\""
      usage
      exit 1
      ;;
  esac
  shift
done


echo "start data monitor for $INPUTDIR"

inotifywait -m -e moved_to -e close_write $INPUTDIR | mbuffer -m 10M |
  while read dir events file
  do
    if [[ $file = *.gz ]]
    then
      echo -n "."
      mv "$INPUTDIR/$file" "$OUTPUTDIR/"
    #else
      #echo "ignoring ${file} ($events)"
    fi
  done
