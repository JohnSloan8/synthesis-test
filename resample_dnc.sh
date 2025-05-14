#!/bin/bash

# Target sample rate
TARGET_SR=22050

# Base directories
DIRS=(
  "audio-files/amanda/correct"
  "audio-files/amanda/wrong"
)

# Loop over each directory
for DIR in "${DIRS[@]}"; do
  echo "Processing directory: $DIR"

  for i in {01..05}; do
    INPUT="$DIR/DNC-abairt-ceart-$i.wav"
    if [[ "$DIR" == *wrong* ]]; then
      INPUT="$DIR/DNC-abairt-micheart-$i.wav"
      OUTPUT="$DIR/DNC-abairt-micheart-$i-resampled.wav"
    else
      OUTPUT="$DIR/DNC-abairt-ceart-$i-resampled.wav"
    fi

    if [ -f "$INPUT" ]; then
      echo "Resampling $INPUT -> $OUTPUT"
      ffmpeg -y -i "$INPUT" -ar $TARGET_SR "$OUTPUT"
    else
      echo "File not found: $INPUT"
    fi
  done
done
