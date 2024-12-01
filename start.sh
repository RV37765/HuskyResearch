#!/bin/bash

# Cleanup
pkill -9 -f vnc
pkill -9 -f websockify
rm -rf /tmp/.X* || true
rm -rf /tmp/.x* || true
rm -rf ~/.vnc/*.log || true
rm -rf ~/.vnc/*.pid || true

# Initialize Xauthority
touch ~/.Xauthority
xauth generate :1 . trusted

# Start VNC server
mkdir -p ~/.vnc
vncserver :1 -geometry 1920x1080 -depth 24 \
    -localhost no \
    -fg \
    -SecurityTypes VncAuth \
    -xstartup /usr/bin/fluxbox &

sleep 5

# Start noVNC
/usr/share/novnc/utils/launch.sh --vnc localhost:5901 --listen 6080 &

# Source ROS environment
source /opt/ros/noetic/setup.bash

# Keep container running
tail -f /dev/null
