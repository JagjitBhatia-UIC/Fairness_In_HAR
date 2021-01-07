from subprocess import PIPE, run

command = ['python', '3D-ResNet/main.py']
result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
with open('3D-ResNet/results/val.json') as f:
    preds = json.load(f)

return preds


