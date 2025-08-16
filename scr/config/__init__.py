#!/usr/bin/python

'''
Created on   : Freelancer August 16th, 2025
@author      : Aryanto
Compiler     : Python 3.12

Version 0.0.1
'''

import  os
import  sys
import  logging         as      Lg
from    configparser    import  ConfigParser
from    ast             import  literal_eval
from    typing          import  Optional, Callable
from    shutil          import  copy as cpfile

TheConfig               = ConfigParser()
BasedDir                = os.path.dirname(os.path.realpath(__file__))
TheConfig.read(os.path.join(BasedDir, 'Configuration.ini'))
Verbose                 = TheConfig.getboolean('DEFAULT', 'VERBOSE', fallback = True)
Levels                  = Lg.INFO if Verbose else Lg.ERROR
LogFormat               = '%(asctime)s - %(name)s: - %(levelname)s [%(lineno)d] - %(message)s'

Lg.basicConfig(handlers = [Lg.StreamHandler(sys.stdout)], 
               level    = Levels,
               format   = LogFormat,
              )
TheLogs                 = Lg.getLogger('[Stuntings]')

def OpenConfList(Set        : str, 
                 ConfigName : str,
                 Config     : Optional[Callable] = None,
                ) :
    Thebiner                = list()
    if Config is None :
        Config              = TheConfig
    try :
        Conf                = Config.get(str(Set), str(ConfigName))
        Thebiner            = literal_eval(Conf)
    except Exception as Arch :
        Warn                = f'''
[OpenConfList ERROR] We failed to open config.
[OpenConfList ERROR] Failure : {Arch}
        '''
        print(Warn)
    finally :
        return Thebiner
