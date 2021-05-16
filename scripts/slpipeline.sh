#! /bin/bash
#cd /content/drive/MyDrive/1pipelinescript && mkdir LCDATA && mkdir TESTLCDATA && mkdir RESERVE && mkdir TESTRESERVE
#cd /content/drive/MyDrive/1pipelinescript && mkdir STORAGE && mkdir TESTSTORAGE #commands to create six folders

TFFOLDER="/content/drive/MyDrive/lczero-training/tf"
YAMLPATH="/content/drive/MyDrive/config/128-10.yaml"

cd $TFFOLDER # to read trainstepslog.txt file with stepsdone number. File should be in this folder
MAINFOLDER="/content" #folder where 6 folders with data are, important for STORAGE<->RESERVE switching when empty
LCDATA="/content/LCDATA" # train chunks folder
TESTLCDATA="/content/TESTLCDATA" # test chunks folder
RESERVE="/content/RESERVE" # folder where used by train.py chunks are moved
TESTRESERVE="/content/TESTRESERVE" # folder where test chunks used by train.py chunks are moved
STORAGE="/content/STORAGE" # folder from where this script moves train chunks to lcdata
TESTSTORAGE="/content/TESTSTORAGE" # folder from where this script moves test chunks to testlcdata
TOTALSTEPS=100000 # planned number of training steps for whole training run
STEPSDONE=$(<trainstepslog.txt) #stepsdone by this script from logfile
TRAININGSTEP=10000 # number of steps equals totalsteps in yaml (one iteration here)
DATASTEPMB=1000 #number in megabytes
TESTDATASTEPMB=60 #number in megabytes, but it will be mirrored in this script verstion according to folders in teststorage
DATAWINDOWMB=6000 #number in megabytes
TESTDATAWINDOWMB=600 #number in megabytes but it will be mirrored in this script version
# To disable mirroring according to data from SLSplit script uncomment 2 function call inside while cycle and 
# comment fragment for mirroring inside mvdata function

DATASTEP=$(( $DATASTEPMB * 1000000 ))
TESTDATASTEP=$(( $TESTDATASTEPMB * 1000000 ))
DATAWINDOW=$(( $DATAWINDOWMB * 1000000 ))
TESTDATAWINDOW=$(( $TESTDATAWINDOWMB * 1000000 ))


echo steps_done $STEPSDONE

function mvdata () {
  # $1 folder from where we move data
  # $2 folder where we move data
  # $3 how many GB of data we move to folder 1000000000 1GB
  cd $1 # directory from where we move folders
  #echo data is moving from $1
  #DSIZE=$(du -sb "$2" | cut -f1) # disc usage of folder where we move data
  echo moving $(( $3 / 1000000 )) MB of data from $1 to $2
  SUM=0
  for FILE in *; 
    do # cycle for moving data into train chunks folder folder by folder
  
    FSIZE=$(du -sb "$FILE" | cut -f1)
    # IFSIZE=$( $DSIZE + $FSIZE )
    #DTSIZE=$(du -sb $DESTINATIONTESTFOLDER | cut -f1)
	  if [[  -d "$FILE" && "$SUM" -lt "$3" ]] # (( "$FSIZE" +  ))
    then 
    mv $FILE $2

    SUM=$(( $SUM + $FSIZE ))
    echo $FILE moved, total $(( $SUM / 1000000 )) MB
      # mv same name test folder code fragment begins
      if [[ $(basename "$2") == "RESERVE" ]]
      then
      OUTF="TESTLCDATA"
      fi
      if [[ $(basename "$2" ) == "LCDATA" ]]
      then
      OUTF="TESTSTORAGE"
      fi
      echo mv $(dirname "$2")"/""$OUTF""/""$FILE" $(dirname "$2")"/""TEST"$(basename "$2" )
      mv $(dirname "$2")"/""$OUTF""/""$FILE" $(dirname "$2")"/""TEST"$(basename "$2" )
      #echo moving_test_folder mv $(dirname "$2")"/""$OUTF""/""$FILE" $(dirname "$2")"/""TEST"$(basename "$2" )
      # mv same name test folder code gragment ends

    fi
    if [[ "$SUM" -ge "$3" ]] 
    then
    echo operation complete
    break
    fi
  done
}

  # $1 folder from where we move data
  # $2 folder where we move data
  # $3 how many GB of data we move to folder 1000000000 1GB
 #tempst="/content/drive/MyDrive/1pipelinescript/lcdata"
 #mvdata $tempst $LCDATA 3000000000

function movein () {
  F2SIZE=$( du -sb "$2" | cut -f1 )
 
  if [[ "$F2SIZE" -lt "$3" ]]
  then 
  mvdata $1 $2 $(( $3 - $F2SIZE ))
  fi
  CHECKSIZE=$( du -sb "$2" | cut -f1 )
  if [[ "$CHECKSIZE" -le $3 ]]; then
  echo $1 is empty # value $4
    if [[ $4 -eq 10 ]]; then # if train folder = true
    cd $MAINFOLDER
    echo RENAMING STORAGE AND RESERVE FOLDERS TO KEEP DATA MOVING IN CYCLE
    mv STORAGE TRANSIT
    mv RESERVE STORAGE
    mv TRANSIT RESERVE
    echo $RESERVE 
    #code fragment for switch when testdata is mirrored
    mv TESTSTORAGE TRANSIT
    mv TESTRESERVE TESTSTORAGE
    mv TRANSIT TESTRESERVE
    #code fragment for switch when testdata is mirrored
    addfirstchar $2 #adding 1 to all folders names inside LCDATA, so they will move out first compared to new cycle data
    addfirstchar $(dirname "$2")"/""TEST"$(basename "$2")    #addfirstchar to "TEST"+LCDATA ($2=path/../LCDATA)
    delfirstchar $(dirname "$2")"/""STORAGE" #deleting possible first 1 from older cycles for STORAGE
    delfirstchar $(dirname "$2")"/""TESTSTORAGE" #deleting possible first 1 from older cycles for TESTSTORAGE
    mvdata $1 $2 $(( $3 - $F2SIZE ))
    fi
    #fragment commented with mirrored train-test, uncomment if needed among with deleting fragment in mvdata
    #if [[ $4 -eq 20 ]]; then # if test folder = true (value 20)
    #cd $MAINFOLDER
    #
    #echo SWITCH IS WORKING  $4
    #mv TESTSTORAGE TRANSIT
    #mv TESTRESERVE TESTSTORAGE
    #mv TRANSIT TESTRESERVE
    #mvdata $1 $2 $(( $3 - $F2SIZE ))
    #fi
  fi
}

function moveout () {
F1SIZE=$( du -sb "$1" | cut -f1 )
echo freeing window $(( $3 / 1000000 )) MB of data moving from $1 to $2
mvdata $1 $2 $3
}
function train() {
cd $TFFOLDER && ./train.py --cfg=$1 # --output=$2 2>&1 #| tee "$ROOT/logs/$(date +%Y%m%d-%H%M%S).log"
  #mv -v $2.pb.gz $NETDIR
  # unbuffer 
}

function addfirstchar () {
#f adds 1 to chunk folders name, so last tr window will be first to move out when new "old" data after switch enter cycle  
#basically this f with deleting 1 f later implements FIFO (1 in first out) for leela training data naming and pipeline script
#$1 first parameter. Should be directory where files are renamed
# addfirstchar $1 to call
cd "$1"
for DIRECTORY in "$1"/*; do
  mv "$DIRECTORY" 1$(basename "$DIRECTORY") #adding 1 to directory name
  done
}

function delfirstchar () {
# delfirstchar $1 to call
INPUTDIR="$1" # folder where all folder will be renamed if needed (first char 1 will be removed)
cd "$1" #parentfolder for mv to work as name changer
for DIR in $INPUTDIR/*; do
  A=$(basename "$DIR")
  FIRSTCHAR=${A:0:1}
  if [[ "$FIRSTCHAR" -eq "1" ]];
  then
  mv "$DIR" "${A:1}"
  fi
done
}


while [ $TOTALSTEPS -ge $STEPSDONE ] 
do
        # from where  #where # how much GB # train or test flag
  moveout $LCDATA $RESERVE $DATASTEP
  #moveout $TESTLCDATA $TESTRESERVE $TESTDATASTEP #not needed with mirroring, thats why commented

  movein $STORAGE $LCDATA $DATAWINDOW 10 
  #movein $TESTSTORAGE $TESTLCDATA $TESTDATAWINDOW 20 #not needed with mirroring, thats why commented
  train $YAMLPATH #$
  STEPSDONE=$(( $STEPSDONE + $TRAININGSTEP ))
  cd $TFFOLDER
  STOPSWITCH=$(<trainstepslog.txt)
  if [[ "$STOPSWITCH" -eq 'stop' ]]; then
  echo "$STEPSDONE" >trainstepslog.txt #writing stepsdone into trainstepslog file
  echo "training stopped"
  break
  fi
  echo $STEPSDONE >trainstepslog.txt #writing stepsdone into trainstepslog file
  #sleep 30
  echo steps_done, $STEPSDONE written to trainstepslog.txt 
done
