#!/bin/bash

start=$1
stop=$2

seq $start $(( stop - 1 )) | parallel --delay 30 ./scripts/preprocess.sh {}
