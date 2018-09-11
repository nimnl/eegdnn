# -*- coding: utf-8 -*- 

import sys, os
import pdb             
import time

#origin = '/home/niago/Documents/PIBITI_2017-2018/Deep_Learning_Code/2018/'
#remote_user = 'geovanens@150.165.75.118'
origin = '/home/geonniago/Documents/PIBITI_2017-2018/Deep_Learning_Code/2018/'
remote_user = 'geovanens@150.165.75.118'
destination = '/home/geovanens/Documentos/COPY'


command_rsync = "rsync -vrh --progress --max-size='20000k'"
full_command = ' '.join([command_rsync, origin, remote_user])
full_command = ':'.join([full_command, destination])

os.system(full_command)

end=time.time()

