import h5py

import silx
import pandas as pd
from silx.io.h5py_utils import File
from cedapp.drx import Calibration
from cedapp.drx import CL_FD_Update as CL


import matplotlib.pyplot as plt
import fabio
scan_lbl='33.1'



loaded_file_DRX=r'/data/visitor/hc6664/id09/20260121/RAW_DATA/Water12/Water12_0001/scan0003/scan_jf1m_0000.h5'
with File(r'/data/visitor/hc6664/id09/20260121/RAW_DATA/Water12/Water12_0001/Water12_0001.h5') as f:
    print(f[scan_lbl]["measurement"].keys())
    t1 = f[scan_lbl]["measurement"]["ch5_time"][:].ravel()
    y1 = f[scan_lbl]["measurement"]["ch5"][:].ravel()
    t2 = f[scan_lbl]["measurement"]["ch6_time"][:].ravel()
    y2 = f[scan_lbl]["measurement"]["ch6"][:].ravel()

if not (len(t1) == len(y1) == len(y2)):
    raise ValueError(
    f"Tailles incohérentes : "
    f"t1={len(t1)}, y1={len(y1)}, y2={len(y2)}"
)

data_oscillo = pd.DataFrame(
    {
    "Time": t1,
    "Channel2": y1,
    "Channel3": y2,
    }
    )
    
plt.plot(t2,y2)
plt.plot(t1,y1)
plt.show()
'''
calib = Calibration.Calib_DRX(
            file_mask=r'/data/visitor/hc6664/id09/20260121/dioptas_files/scan_jf1m_0000.mask',
            file_poni=r'/data/visitor/hc6664/id09/20260121/dioptas_files/calib_ddac2_correct.poni',
            theta_range=[8,30],
            energy=19000,
        )


img_data=fabio.open(loaded_file_DRX)


tth, intens = Calibration.Integrate_DRX(img_data.getframe(50).data, calib.mask, calib.ai, theta_range=calib.theta_range)

print('start')
CEDX = CL.CED_DRX(loaded_file_DRX
    ,
    calib,
    19000,
    data_oscillo=data_oscillo,
    time_index="Channel2"
)
CEDX.Print(Oscilo=True)

'''
