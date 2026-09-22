import subprocess
import sys
import os

repo_dir = r"C:\Users\ASUS TUF\Desktop\tts-mini\tts-mini"
python_exe = os.path.join(repo_dir, ".venv", "Scripts", "python.exe")
log_out = os.path.join(repo_dir, "training_acoustic_resumed.log")
log_err = os.path.join(repo_dir, "training_acoustic_resumed.err.log")

cmd = [
    python_exe,
    "-u",  # unbuffered stdout/stderr
    "train_acoustic.py",
    "--data_dir", "km_kh_male",
    "--epochs", "100",
    "--resume"
]

with open(log_out, "a", encoding="utf-8") as out_f, open(log_err, "a", encoding="utf-8") as err_f:
    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=out_f,
        stderr=err_f,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    )

print(f"Acoustic training started with PID: {proc.pid}")
