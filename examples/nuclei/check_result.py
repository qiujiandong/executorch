#!/usr/bin/env python3

import subprocess
import re
from nuclei_example import print_result, label

output = subprocess.run("make SIMU=qemu clean all run_qemu", shell=True, capture_output=True)

result = [[float(x) for x in re.findall(r"-?\d\.\d{6}" , str(output.stdout))]]
print_result(result, label)
