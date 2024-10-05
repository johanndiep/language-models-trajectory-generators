#!/bin/bash

# Check for necessary packages and install if missing
sudo apt-get update

# Install X11 utilities if not already installed
sudo apt-get install -y x11-apps

# Enable X11 forwarding in the SSH server config
sudo sed -i 's/^#X11Forwarding no/X11Forwarding yes/' /etc/ssh/sshd_config
sudo sed -i 's/^#X11UseLocalhost yes/X11UseLocalhost yes/' /etc/ssh/sshd_config

# Restart SSH service to apply changes
sudo service ssh restart

# Confirm that X11 forwarding is enabled
echo "X11 forwarding setup completed. SSH configuration updated, and SSH service restarted."

# Display current SSH config for verification
grep -E "X11Forwarding|X11UseLocalhost" /etc/ssh/sshd_config
