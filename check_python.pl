user='ml0001' 
pw ='7902$*OPidjid' 
dsn ='db20210112_low' 
import cx_Oracle
print('Connection (user pw dsn):', user, pw, dsn) 
connection = cx_Oracle.connect(user, pw, dsn) 
print('DB version:', connection.version)
cursor = connection.cursor() 
sql = "select metric_name from sys.gv_$con_sysmetric" 
cursor.execute(sql) 
print( cursor.fetchmany(5) )
cursor.close() 
connection.close()
