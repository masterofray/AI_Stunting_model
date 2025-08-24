#!/usr/bin/python

'''
Created on   : Freelancer August 23rd, 2025
@author      : Aryanto
Compiler     : Python 3.12

Version 0.0.1
'''

from psycopg2 import DatabaseError
from psycopg2.extensions import STATUS_READY

def PostGre_Insert(Data       : pd.DataFrame,
                   TableName  : str,
                   Connection : object,
                  ) -> bool:
    if Connection is None:
        print("Error: The database connection is not open.")
        return False

    values = list(Data.itertuples(index=False, name=None))
    Kolom = ', '.join([f'"{c}"' for c in Data.columns])
    try:
        with Connection.cursor() as postgr:
            if Connection.closed or Connection.status != STATUS_READY:
                print("Transaction is in an aborted state. Rolling back.")
                Connection.rollback()

            print(f"Dropping table '{TableName}' if it exists.")
            postgr.execute(f"DROP TABLE IF EXISTS \"{TableName}\"")

            print(f"Creating table '{TableName}'.")
            tipedata = {'int64'       : 'BIGINT',
                        'int32'       : 'INTEGER',
                        'float64'     : 'DOUBLE PRECISION',
                        'float32'     : 'REAL',
                        'object'      : 'TEXT',
                        'bool'        : 'BOOLEAN',
                        'datetime64[ns]' : 'TIMESTAMP',
                        'timedelta[ns]'  : 'INTERVAL',
                        'category'    : 'VARCHAR(255)',
                        'complex64'   : 'TEXT',
                        'complex128'  : 'TEXT',
                        'string'      : 'TEXT',
                       }
            
            Quote = list()
            for column, dtype in Data.dtypes.items():
                sql_type = tipedata.get(str(dtype), 'TEXT')
                Quote.append(f'"{column}" {sql_type}')

            Query = f"""CREATE TABLE "{TableName}" (
                        {', '.join(Quote)}
            );
            """
            postgr.execute(Query)

            print(f"Inserting data into '{TableName}'.")
            placeholders = ','.join(['%s'] * len(Data.columns))
            SQLdata = f"INSERT INTO \"{TableName}\"({Kolom}) VALUES({placeholders})"
            postgr.executemany(SQLdata, values)
        
        Connection.commit()
        print("Table created and data inserted successfully.")
        Success = True

    except DatabaseError as Arc:
        print(f"A database error occurred: {Arc}")
        Connection.rollback()
        Success = False

    finally:
        print('Transaction complete.')
        return Success

if __name__ == '__main__':
    Status = PostGre_Insert(Data       = districtID, 
                            TableName  = 'DistrictID', 
                            Connection = PostConnected,
                           )
    print(f'\n\nHasil pekerjaan : {Status} ')
