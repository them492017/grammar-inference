import os
import subprocess

# Create a virtual environment
venv_dir = "myenv"
subprocess.run(["python", "-m", "venv", venv_dir])

# Activate the virtual environment
activate_script = os.path.join(venv_dir, "bin", "activate")
activate_command = f"source {activate_script}"

urls = [
    "https://rahul.gopinath.org/py/simplefuzzer-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/rxfuzzer-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/earleyparser-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/cfgrandomsample-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/cfgremoveepsilon-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/gatleastsinglefault-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/hdd-0.0.1-py2.py3-none-any.whl",
    "https://rahul.gopinath.org/py/ddset-0.0.1-py2.py3-none-any.whl",
]
# Install wheels
for url in urls:
    install_command = f"{activate_command} && pip install {url}"
    subprocess.run(install_command, shell=True)

print("All wheels have been installed in the virtual environment.")
