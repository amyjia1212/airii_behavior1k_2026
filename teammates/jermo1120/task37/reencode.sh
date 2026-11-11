#!/bin/bash
# Re-encode task 37 videos to short-GOP H.264 (matches the original run: CRF 18, -g 4).
# Replaces a file only if the frame count is unchanged.
set -u
f="$1"
count() { ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$1"; }
codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f")
[ "$codec" = "h264" ] && exit 0
tmp="${f%.mp4}.h264.tmp.mp4"
ffmpeg -nostdin -loglevel error -y -i "$f" -c:v libx264 -preset fast -crf 18 -g 4 -pix_fmt yuv420p -an "$tmp" || { echo "FAIL encode $f"; rm -f "$tmp"; exit 1; }
a=$(count "$f"); b=$(count "$tmp")
if [ "$a" = "$b" ]; then mv "$tmp" "$f"; else echo "FAIL count $f $a != $b"; rm -f "$tmp"; exit 1; fi
