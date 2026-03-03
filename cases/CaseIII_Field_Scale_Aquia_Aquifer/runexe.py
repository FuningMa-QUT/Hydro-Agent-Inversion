import os
import numpy as np
import time
def run_exe():
    cmd_root='echo.|SOWCOM_V2_EOS9.exe'
    os.system(cmd_root)
    if not os.path.exists('.\\run_exe'):
        os.mkdir('.\\run_exe')
    # a=time.time()
    # np.savetxt('.\\run_exe\\{}.dat'.format(a),np.array([1]))
    # print('Runing Forward Modeling')
    return
if __name__=="__main__":
    run_exe()
