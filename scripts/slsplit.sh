#! /bin/bash
# This simple script with just 3 parameters will move 10 percent of chunks from untarred train data chunks folders
# to the test chunks folders with the same name. Script moves approximately 1 gb of data per minute on hdd and old cpu.
INPUTF="/content/drive/MyDrive/1pipelinescript/LCDATA"
OUTPUTF="/content/drive/MyDrive/1pipelinescript/TESTLCDATA"
TESTRATIO=10 #every 10th file will move to outputF, randomly. For 0,9 train ratio parameter equals 10, 20 for 0,95 train ratio etc/
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
