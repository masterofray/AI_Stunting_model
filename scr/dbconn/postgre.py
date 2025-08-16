#!/usr/bin/python

'''
Created on   : Freelancer August 16th, 2025
@author      : Aryanto
Compiler     : Python 3.12

Version 0.0.1
'''

import sys
import pandas   as pd
from   psycopg2 import connect, DatabaseError
from   typing   import Dict
from   pathlib  import  Path
from   pdb      import  set_trace

Locdir = Path(__file__).resolve()
sys.path.append(str(Locdir.parents[0]))
from dbconn import config

sys.path.append(str(Locdir.parents[1]))
from config import  (TheLogs, TheConfig)

def MakeConn(parameter : Dict = None) -> object:
    try:
        if not parameter:
            parameter = {"host"    : TheConfig.get('POSTGRE', 'HOST_IP'),
                         "database": TheConfig.get('POSTGRE', 'DB_NAME'),
                         "user"    : TheConfig.get('POSTGRE', 'DB_USER'),
                         "password": TheConfig.get('POSTGRE', 'DB_PWD'),
                        }
        PostConnected = connect(**parameter)
    except DatabaseErrror as arch:
        print(f'Got error : {arch}')
        PostConnected = None
    return PostConnected

def GetData(Koneksi   : object,
            query     : str,
            usepandas : bool = False,
           ) -> pd.DataFrame:
    assert len(query) > 10, 'Not sufficient query!'
    if not usepandas :
        postgr = Koneksi.cursor()
        postgr.execute(query)
        raw = postgr.fetchall()
        header = [item[0] for item in postgr.description]
        data = pd.DataFrame(raw, columns = header)
        postgr.close()
    else:
        data = pd.read_sql(query, Koneksi)
    return data

if __name__ == '__main__':
    pass