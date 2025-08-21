
FROM osrf/ros:noetic-desktop-full

# Install VNC, window manager, and additional dependencies
RUN apt-get update && apt-get install -y \
    tigervnc-standalone-server \
    tigervnc-common \
    fluxbox \
    wget \
    net-tools \
    novnc \
    python3-pip \
    xterm \
    dbus-x11 \
    x11-utils \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Husky packages
RUN apt-get update && apt-get install -y \
    ros-noetic-husky-simulator \
    ros-noetic-husky-navigation \
    ros-noetic-husky-viz \
    && rm -rf /var/lib/apt/lists/*

# Set up VNC password
RUN mkdir -p ~/.vnc && \
    echo "123" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

# Set up Xauthority
RUN touch ~/.Xauthority

# Add startup script
COPY start.sh /
RUN chmod +x /start.sh

EXPOSE 5901
EXPOSE 6080

COPY launch_and_move3.py /
RUN chmod +x /launch_and_move3.py

CMD ["/start.sh"]

