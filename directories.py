import subprocess
import sys

with open("requirements.txt", "r") as file:
    packages = file.readlines()

for package in packages:
    package = package.strip()

    if package:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package
        ])

print("\n✅ All required python packages are installed!")