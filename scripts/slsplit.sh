#! /bin/bash
# This simple script with just 3 parameters will move 10 percent of chunks from untarred train data chunks folders
# to the test chunks folders with the same name. Script prepares approximately 1 gb of data per minute on hdd and old cpu.
# command example to make .sh file executable: cd /content/drive/MyDrive/1pipelinescript/ && chmod +x slsplit.sh

#INPUTF="/content/drive/MyDrive/1pipelinescript/LCDATA"
#OUTPUTF="/content/drive/MyDrive/1pipelinescript/TESTLCDATA"
INPUTF="$1" #first parameter passed to script during call. Example: cd /content/drive/MyDrive/1pipelinescript/ && ./slsplit.sh /content/STORAGE /content/TESTSTORAGE
OUTPUTF="$2" #second parameter passed to script during call. Result after parsing $1 $2 parameters: randomply moving 10 percent of chunks from /content/STORAGE to /content/TESTSTORAGE
TESTRATIO=10 #every 10th file will move to outputF, randomly. For 0,9 train ratio parameter equals 10, 20 for 0,95 train ratio etc/
#TESTRATIO="$3" # will be 3rd parameter during .sh launch from shell when uncommented
function makedirectory () {
cd $OUTPUTF && mkdir $1 # creating directory with $1 FOLDER name
}
echo TRAIN TEST DATA SPLITTING SCRIPT
echo Randomly moving $TESTRATIO percent of chunks from $INPUTF 
echo to $OUTPUTF
for FOLDER in $INPUTF/*; do
    makedirectory $(basename "$FOLDER")  # $(basename $(dirname "/path/...") )
    echo folder_in_process $FOLDER
    for FILE in "$FOLDER"/*; 
    do
        if [[ $((1 + "$RANDOM" % "$TESTRATIO")) -eq "$TESTRATIO" ]]
        then 
        mv "$FILE" "$OUTPUTF""/"$( basename "$FOLDER")
        fi
    done
done
