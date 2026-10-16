import os
import time
import platform

# Wait for 1 hour (3600 seconds)
time.sleep(3200)

# Detect OS and run shutdown command
if platform.system() == "Windows":
    os.system("shutdown /s /t 1")
elif platform.system() == "Linux" or platform.system() == "Darwin":  # Darwin = macOS
    os.system("shutdown -h now")
else:
    print("Unsupported OS for shutdown")
