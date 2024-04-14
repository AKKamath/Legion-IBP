#!/bin/bash

while [[ $( grep "ready for serving" $1 ) == "" ]]
do
        sleep 1
        #echo "Waiting for graph setup"
done

echo "Training Ready"
