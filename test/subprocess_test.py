import subprocess

result = subprocess.run(['dir'], stdout=subprocess.PIPE, shell=True)
print(result.stdout.decode('latin1'))