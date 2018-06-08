#!/bin/bash

NETDIR="/home/folkert/data/run1"
TSTDIR="/home/folkert/data/work/test"
TRNDIR="/home/folkert/data/work/train"

cd /home/folkert/data/work
echo "monitor start"

n=0
cn=100
lval=6553

ntst=0
ntrn=0
ntsttot=0
ntrntot=0


inotifywait -m $NETDIR -e create -e moved_to | mbuffer -m 10M |
    while read path action file; do

        if [ "$action" = "MOVED_TO" ]
        then
                let "n++"

                #echo "The file '$file' appeared in directory '$path' via '$action'"
                sha=$(sha256sum $NETDIR/$file | awk '{print $1}' | cut -c1-4)
                val=$((16#$sha))

                fn=$NETDIR/$file

                if [ $val -lt $lval ]
                then
                        echo -n "*"
                        let "ntst++"
                        let "ntsttot++"
                        target=$TSTDIR/$file
                else
                        echo -n "T"
                        let "ntrn++"
                        let "ntrntot++"
                        target=$TRNDIR/$file
                fi

                if [ ! -f $target ]
                then
                        cp --preserve=timestamps $fn $target
                fi
        else
                if [ "$action" = "CREATE" ]
                then
                        ec=$(echo " ")
                else
                        echo "The file '$file' appeared in directory '$path' via '$action'"
                fi
        fi

        if [ $n -eq $cn ]
        then
                echo " - $(date) - $ntst $ntrn $ntsttot $ntrntot - $file"

                n=0
                ntst=0
                ntrn=0
        fi

    done

