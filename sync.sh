#!/bin/bash

TARGET_HOST=
TARGET_USER=ubuntu
PROJECT=PSGD

TARGET_PATH=/home/$TARGET_USER/$PROJECT
LOCAL_PATH=
KEY_PATH=

rsync -avhP \
  -e "ssh -i $KEY_PATH" \
  --include 'src' \
  --exclude 'sync.sh' \
  --exclude '.idea' \
  --exclude 'src/numa.h' \
  --exclude 'data' \
  --exclude 'bin/*' \
  --exclude 'analyse/' \
  --exclude 'results/*' \
  --include 'input.txt' \
  $LOCAL_PATH $TARGET_USER@$TARGET_HOST:$TARGET_PATH


TARGET_PATH=/home/$TARGET_USER/$PROJECT/results
LOCAL_PATH=/Users/Maksim.Zuev/Documents/$PROJECT
rsync -avhP \
  -e "ssh -i $KEY_PATH" \
  $TARGET_USER@$TARGET_HOST:$TARGET_PATH $LOCAL_PATH
