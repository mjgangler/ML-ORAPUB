#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection System (ML1)
# (c)2021 OraPub, Inc - All rights reserved.
# WARNING: Use at your own risk.
# 
# For what has changed in this version, see the what-has-changed.txt document.
# 
# Check DB Connection and Query: $ python ./AnomDetect-1.py db
# 
# Check Email: $ python ./AnomDetect-1.py email
# 
# Is it working? Watch it work. $ python -u ./AnomDetect-1.py
# 
# Deploy. $ nohup python -u ./AnomDetect-1.py >out.txt 2>out.txt &

# In[1]:


lastCodeUpdate = '1-April-2021' # ref: what-has-changed.txt

# What has changed?
#   There are command line checks for either 'db' or 'email'
#   Detects changes in DB credentials and then uses the credentials... without stoping the AD system from running.
#   If unable to connect or query DB, then refreshes DB credentials, sleeps, then continues main loop
#   data_max_samples_modeled is working. The raw collected data is never archieved or removed. 
#     the X most recent rows are used in the model. This means, you can reduce the number of rows modeled (up or down) as desired.
#   There are now two models available, kmeans and Isoluation Forest. They are very different, though the plot can look very similar, because it is... only the score changes, not the point.
#   You can change the size of the charts and the point size.
#   The charts now show all anomalous points in magenta with the current sample in green or red.
#     This allows you to see where and what the model considers anomalous... great for adjusting and learning.
#   You can change the orientationof the 3D chart.
#   You can set the dimentional reduction model (PCA, ICA) in the configuration file.
#   All configuration file changes take affect just before the next sample. So, you can experiment.


# In[2]:


Testing    = False
InNotebook = True
CmdCheck   = '' # Only used if InNotebook, then normally, set to '' or to do a check, set to either 'db' or 'email'


# In[3]:


# Directory Settings

import sys
import os

if not InNotebook:
    try:
        baseDir    = str(sys.argv[1]) + '/'
        configFile = str(sys.argv[2])
    except:
        baseDir    = os.getcwd() + '/'
        configFile = 'AnomDetect.cfg'
else:
    baseDir    = os.getcwd() + '/'
    configFile = 'AnomDetect.cfg'

chartsDir2D = baseDir + 'charts2D'   # <-------- Make sure this directory exists!!!
chartsDir3D = baseDir + 'charts3D'   # <-------- Make sure this directory exists!!!
alertFN     = baseDir + 'alertlog.txt'

print()
print('Directories and files:')
print('  baseDir    ', baseDir)
print('  chartsDir2D', chartsDir2D)
print('  chartsDir3D', chartsDir3D)
print('  configFile ', configFile)
print('  alertFN    ', alertFN)


# In[4]:


def doExit():
    import sys
    print('Exiting clean')
    try:    
        cursor.close()
        dbConnection.close()
    except:
        pass
    
    sys.exit() 


# In[5]:


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# In[6]:


def alertLogWrite(line_in):

    from datetime import datetime

    now = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    f = open(alertFN,'a')
    f.write(now + ', ' + line_in + '\n')
    
if Testing:
    x = 'Yo Craig, This is an alert log test.'
    alertLogWrite(x)


# In[7]:


def checkDBFileChange():

    # Check if DB access credentials have changed in the AnomDetect_DB.py file.
    
    change = False
    
    import AnomDetect_DB as db
    import importlib
    
    current_user = db.user
    current_pw   = db.pw
    current_dsn  = db.dsn
    
    importlib.reload(db)
    
    if (current_user != db.user or current_pw != db.pw or current_dsn != db.dsn):
        change = True
    else:
        change = False
    
    return(change)

if Testing:
    print('Testing Function: checkDBFileChange')
    print(checkDBFileChange())
    


# In[8]:


def readParamFile(verbose_in):
    
    # ref: https://zetcode.com/python/configparser
    
    import configparser
    
    config = configparser.ConfigParser()
    
    fullConfigFile = baseDir + configFile
    config.read(fullConfigFile)
    
    try:
        prior = parCore
    except:
        prior = config['core']
    
    myParCore = config['core']
    
    pars = str({section: dict(config[section]) for section in config.sections()})
    
    if verbose_in:
        print('\nParameter Settings:\n')
        print(pars, '\n')
        alertLogWrite('Current Parameters, ' + pars)
    
    if prior != myParCore:
        alertLogWrite('Detected parameter file change, ' + pars)
        print('\nParemeter Setting CHANGE:\n')
        print(pars, '\n')
    
    return(myParCore)

if Testing:
    print('Testing Function: readParamFile')
    # pareCore is outside of the function, so results available everywhere
    parCore = readParamFile(True) # keep result as parCore, so it can be used when testing
    


# In[9]:


def sendAlertEmail_wo_charts(subject_in, message_in, verbose_in):
    
    # Ref: https://realpython.com/python-send-email/
    
    result = False

    import AnomDetect_Email as email
    import smtplib
    import ssl
    import importlib
    
    importlib.reload(email)

    port           = email.port
    smtp_server    = email.smtp_server
    sender_email   = email.sender_email
    sender_pass    = email.sender_pass
    receiver_email = email.receiver_email

    fullMessage = "Subject: " + str(subject_in) + "\n\n" + message_in

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, receiver_email, fullMessage)
        result = True
    except:
        print('Email alert message failed')

    return(result)

# To perform the send test, remove both sets of the 3 double quotes below
"""
if Testing:
    print('Testing Function: sendAlertEmail \n')
    from datetime import datetime
    now = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    mySubject = 'AD1 anomaly detected at ' + now
    myMessage = 'Hey Craig, This is an email TEST at ' + now
    
    if str2bool(parCore['alert_email_wo_charts']):
        q = sendAlertEmail_wo_charts(mySubject, myMessage, True)
        print('q',q)
    else:
        print('Alert email WO CHARTS is not enabled.')
"""

print('Done.')


# In[10]:


def sendAlertEmail_w_charts(subject_in, message_in, filename_in, verbose_in):
    
    # Ref: https://realpython.com/python-send-email/
    
    # filename_in is NOT the full path, just the image filename
    
    result = False
    
    import importlib
    import AnomDetect_Email as e
    importlib.reload(e)
    
    import email, smtplib, ssl
    
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
        
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = e.sender_email
    message["To"] = e.receiver_email
    message["Subject"] = subject_in
    
    # Add body to email
    message.attach(MIMEText(message_in, "plain"))
    
    for myDir in [chartsDir2D, chartsDir3D]:
        
        # Open graphic file in binary mode
        with open(myDir + '/' + filename_in, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)
        
        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename_in}",
        )
        
        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()
    
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(e.smtp_server, e.port, context=context) as server:
        server.login(e.sender_email, e.sender_pass)
        server.sendmail(e.sender_email, e.receiver_email, text)
    result = True
    
    return(result)

# To perform the send test, remove both sets of the 3 double quotes below
"""
if Testing:
    print('Testing Function: sendAlertEmail WITH CHARTS \n')
    from datetime import datetime
    now = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    mySubject = 'AD1 anomaly detected at ' + now
    myMessage = 'Hey Craig, This is an email TEST at ' + now + ' with CHARTS!'
    
    try:
        if str2bool(parCore['alert_email_w_charts']):
            myFN = "20210323072349-False.png"
            q = sendAlertEmail_w_charts(mySubject, myMessage, myFN , True)
            print('q',q)
        else:
            print('Alert email WITH CHARTS is not enabled.')
    except:
        print('For test to work, you need an existing filename...sorry.')
"""

print('Done.')


# In[11]:


def makeDbConnection(show_error_in, verbose_in):
    
    import AnomDetect_DB as db
    import cx_Oracle
    import os
    import importlib
    
    importlib.reload(db)
    
    conTF = False
    
    if verbose_in:
        print('Attempting Oracle connection db.user db.dsn ', db.user, db.dsn, end=' ...')
    
    try:
        conDetails = cx_Oracle.connect(db.user, db.pw, db.dsn)
        
    except cx_Oracle.Error as e:
        conDetails = 0
        
        if show_error_in:            
            errorObj, = e.args
            print('failed to connect.')
            print('   Error Code       :', errorObj.code)
            print('   Error Message    :', errorObj.message)
            print('   TNS_ADMIN        :', os.getenv('TNS_ADMIN'))
            print('   PATH             :', os.getenv('PATH'))
            print('   DYLD_LIBRARY_PATH:', os.getenv('DYLD_LIBRARY_PATH'))
    
    else:
        conTF = True
        if verbose_in:
            print('success.')
            print('Connected Oracle version', conDetails.version)
        
    return( conTF, conDetails )


if Testing:
    print('Testing Function: makeDbConnection \n')
    
    print('Test 1')
    result, dbConnection = makeDbConnection(show_error_in=True,verbose_in=True)
    print('Test 1 result', result, 'dbConnection', dbConnection)
    
    print('\nTest 2')
    result, dbConnection = makeDbConnection(show_error_in=True,verbose_in=False)
    print('Test 2 result', result, 'dbConnection', dbConnection)
 
    print('\nDone.')


# In[12]:


def querySourceCheck(dbConnection_in, verbose_in):
    
    import cx_Oracle
    
    myResult = False
    
    if verbose_in:
        print('Attempting to query from data source')

    try:
        cursor = dbConnection_in.cursor()
    
    except cx_Oracle.Error as e:
        errorObj, = e.args
        print('failed to query.')
        print('   Error Code       :', errorObj.code)
        print('   Error Message    :', errorObj.message)

    else:
        sql = "select distinct(metric_name) from " + parCore['oracle_perf_data_table_name']               + " " + parCore['oracle_perf_data_where_clause'] + " order by metric_name"
        if verbose_in:
            print('sql', sql)
        
        try:
            cursor.execute(sql)
            #print( cursor.fetchmany(5) )
            if verbose_in:
                print( cursor.fetchall())
            
            myResult = True
        
        except:
            myResult = False
            
    cursor.close()
    
    return(myResult)

if Testing:
    print('Testing Function: querySourceCheck')
    
    result, conTest = makeDbConnection(show_error_in=True,verbose_in=True)
    print("\nTest 1 True")
    print( querySourceCheck(conTest, True) )

    print("\nTest 2 False")
    print( querySourceCheck(conTest, False) )


# In[13]:


def loadData(fn_in, verbose_in):
    
    import pandas as pd
    
    resultTF = False
    resultDF = pd.DataFrame()
    
    if verbose_in:
        print("Loading data "+ str(fn_in), end=" ... ")
    
    try:
        resultDF = pd.read_csv(fn_in, low_memory=False)
    
    except:
        if verbose_in:
            print('not found.')
    
    else:
        resultTF = True
        resultDF.columns = map(str.lower, resultDF.columns) # all features to be in lower case
        resultDF.fillna(0) # replace NaN/Nulls with zero
        
        if verbose_in:
            print("done. shape", resultDF.shape)
    
    return(resultTF, resultDF)

if Testing:
    print('Testing Function: loadData \n')
    
    print('Test 1 - Should succed')
    x = 'http://filebank.orapub.com/DataSets/quick_restart-1.csv'
    result, xDF = loadData(x, True)
    if result:
        print('xDF.shape', xDF.shape)
    
    print('\nTest 2 - Should fail')
    x = 'http://filebank.orapub.COMMM/DataSets/quick_restart-1.csv'
    result, xDF = loadData(x, True)
    if result:
        print('xDF.shape', xDF.shape)
    
    print('\nDone.')


# In[14]:


def loadRestartData(verbose_in, showTime_in):
    
    import pandas as pd
    import datetime
    
    import time
    t0 = time.perf_counter()
    
    resultTF          = False
    resultDF          = pd.DataFrame()
    oldestRowDatetime = 0
    newestRowDatetime = 0
    oldestDatetime    = 0
    newestDatetime    = 0
        
    if parCore['data_restart_enabled']:
        
        csvFN  = parCore['data_restart_csv_file']
        httpFN = parCore['data_restart_http_base']
        
        if verbose_in:
            print('Loading CSV', csvFN, end=" ... ")
        resultTF, resultDF = loadData(csvFN,verbose_in)
        
        if not resultTF:
            if verbose_in:
                print('\nLoading http', httpFN + '/' + csvFN)
            resultTF, resultDF = loadData(httpFN + '/' + csvFN,verbose_in)
        
        if not resultTF:
            print('No restart data (csv, http) found.')
        else:
            resultDF.columns = map(str.lower, resultDF.columns) # all features to be in lower case
            x = resultDF.dt.head(1)
            oldest = x.to_string(index=False)
            x = resultDF.dt.tail(1)
            newest = x.to_string(index=False)
            
            if verbose_in:
                print('oldest', type(oldest), oldest)
                print('newest', type(newest), newest)
            
            oldestDatetime = datetime.datetime.strptime(oldest, '%Y%m%d%H%M%S')
            #print('oldestDatetime', oldestDatetime)
            newestDatetime = datetime.datetime.strptime(newest, '%Y%m%d%H%M%S')
            #print('newestDatetime', newestDatetime)
            if verbose_in:
                print('oldestDatetime newestDatetime', oldestDatetime, newestDatetime, end='')
        
        if showTime_in:
            print(f'(LR et {time.perf_counter() - t0:0.1f}s)')
        else:
            print()
    else:
        if verbose_in:
            print('Data restart is not enabled.')

    return(resultTF,resultDF, oldestDatetime, newestDatetime)

if Testing:
    print('Testing Function: loadRestartData')
    
    gotDataTF, xDF, bDt, eDt = loadRestartData(True, True)
    if gotDataTF:
        print('\nDoublecheck')
        print('xDF.shape', xDF.shape)
        print('bDt, eDt', bDt, eDt)


# In[15]:


def checkConnectAndQuery(verbose_in):
    
    result = False

    conResult, mydbcon = makeDbConnection(show_error_in=True,verbose_in=verbose_in)
    
    if conResult:
        result  = querySourceCheck(mydbcon, verbose_in)
    
    return(result)

if Testing:
    print('Testing Function: checkConnectAndQuery \n')

    print('Test 1 - True verbose, should return True')
    print( checkConnectAndQuery(True) )
    
    print('\nTest 2 - False verbose, should return True')
    print( checkConnectAndQuery(False) )
    
    print('\nDone.')


# In[16]:


def getMaxDate(dataType_in, dataSource_in, verbose_in):
    
    # Only the date is returned, not the metrics
    
    if dataType_in == 'dataframe':
        if len(dataSource_in) > 0:
            # if the below astype is not used, an error returned related to types
            maxDate = dataSource_in['dt'].astype('int64').max()
        else:
            maxDate = 0
    
    elif dataType_in == 'oracle':
        try:
            # dataSource_in is not used
            sql = "select max(to_number(to_char(begin_time, 'YYYYMMDDHH24MISS'))) dt from "                   + parCore['oracle_perf_data_table_name'] + " "                   + parCore['oracle_perf_data_where_clause']

            cursor = dbConnection.cursor()
            cursor.execute(sql)
            row = cursor.fetchone()
            cursor.close()
            maxDate = row[0]
        except:
            print('Unable to perform DB query. Function getMaxDate.')
            maxDate = 0
    
    else:
        if verbose_in:
            print('getMaxDate: Invalid dataType_in, returning 0')
        maxDate = 0
    
    if verbose_in:
        print('getMaxDate return:', maxDate)
        
    return(maxDate)
    
if Testing:
    print('Testing Function: getMaxDate')
    
    import pandas
    
    print('Test 1 - Should return data. Test: normal ora')
    print( getMaxDate('oracle','na', True) )
    
    print('\nTest 2 - Should NOT return data. Test: empty DF')
    myCols = ['name', 'dt']
    df = pandas.DataFrame(columns = myCols)
    print( getMaxDate('dataframe',df, True) )
    
    print('\nTest 3 - Should return data. Test: normal DF')
    data = [['tom', 10], ['nick', 15], ['juli', 14]] 
    df = pandas.DataFrame(data, columns = ['name', 'dt'])
    print( getMaxDate('dataframe',df, True) )
    
    print('\nTest 4 - Should NOT return data. Test: bogus data type param')
    df = pandas.DataFrame(columns = ['name', 'dt'])
    print( getMaxDate('woops',df, True) )
    
    print('\nTest 5 - Should NOT return data. Test: empty DF')    
    df = pandas.DataFrame(columns=myCols)
    currentMaxDate = getMaxDate('dataframe', df, True)
    print('currentMaxDate', currentMaxDate)


# In[17]:


def getNewData(lastDataBeginTime_in, verbose_in):
    
    # if zero rows returned, then no new data
    
    sql = "select to_char(begin_time, 'YYYYMMDDHH24MISS') dt, metric_name, value metric_value " +           "from   " + parCore['oracle_perf_data_table_name'] + " " +            parCore['oracle_perf_data_where_clause'] +                 " and  to_char(begin_time, 'YYYYMMDDHH24MISS') > :begTime"
    
    lastDataBeginTime = lastDataBeginTime_in
    
    try:
        cursor = dbConnection.cursor()
        cursor.execute(sql, begTime = str(lastDataBeginTime_in))  # setting the bind variable
        newRows = cursor.fetchall()
        cursor.close()
        #print('len(newRows)', len(newRows))
    except:
        newRows = []
        print('Unable to query new data. Function, getNewData')
    
    # if zero rows returned, then no new data
    if len(newRows) == 0:
        lastDataBeginTime = 0
    else:
        lastDataBeginTime = newRows[0][0]
    
    return(lastDataBeginTime, newRows)


if Testing:
    print('Testing Function: getNewData')
    
    print('Test 1 - should return data')
    dt, rows = getNewData('198901010101', True)
    print(dt, rows)

    print('\nTest 2 - should not return data')
    print(getNewData(dt, True))


# In[18]:


def resetCoreDFtypes(DF_in, verbose_in):
    
    #if verbose_in:
    #    print('In:')
    #    print(DF_in.dtypes)

    DF_in['dt']           = DF_in['dt'].astype('int64')
    DF_in['metric_name']  = DF_in['metric_name'].astype('str')
    DF_in['metric_value'] = DF_in['metric_value'].astype('float')
    
    #if verbose_in:
    #    print('Out:')
    #    print(DF_in.dtypes)
    
    return(DF_in)


# In[19]:


def addNewRows(coreDF_in, verbose_in):
        
    result = False
    availableMaxDate = 0
    
    try:
        #print('A')
        # Get most recent date for the data collected and data available
        currentMaxDate    = getMaxDate('dataframe', coreDF_in, verbose_in)
        #print('B')
        
        availableMaxDate  = getMaxDate('oracle','na', verbose_in)
        
        # If new is available, get it
        #print('C')
        
        if int(availableMaxDate) > int(currentMaxDate):
            #print('D')
            
            newMaxDate, newRows = getNewData(currentMaxDate, verbose_in)
            newRowsDF           = pd.DataFrame(newRows, columns=coreDF_in.columns)
            #print('E')
        
        else:
            newRowsDF           = pd.DataFrame(columns=coreDF_in.columns)
            #print('F')
        
        # Set data types
        coreDF_in = resetCoreDFtypes(coreDF_in, verbose_in)
        #print('G')
        
        # Add new data to existing data
        appendedDF = coreDF_in.append(newRowsDF)
        #print('H')
        
        # Always print this... it's like a heartbeat
        #
        print('rawDF', coreDF_in.shape, 'newRowsDF', newRowsDF.shape, 'appendedDF', appendedDF.shape, end='')

        if verbose_in:
            #print(type(appendedDF))

            # https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
            x = appendedDF[appendedDF['metric_name'] == 'Average Active Sessions'].metric_value.tail(1)
            print(' AAS', x.to_string(index=False))
        
        result = True
    except Exception as e:
        print('\nException. Function: addNewRows:',e)
        appendedDF = coreDF_in.copy()
        print('\nUnable to retreive new data. Function, addNewRows 1\n')
    
    if availableMaxDate == 0:
        result = False
        appendedDF = coreDF_in.copy()
        print('\nUnable to retreive new data. Function, addNewRows 2\n')
        
    # Return the combined existing and new data
    return(result, appendedDF)

if Testing:
    print('Testing Function: addNewRows')
    
    mySleepSec = 0
    
    import pandas as pd
    myCols=['dt','metric_name','metric_value']
    qDF = pd.DataFrame(columns=myCols)
    
    print('Test 1 - Should return data but no Verbose\n')
    qResult, qDF = addNewRows(qDF, False)
    print('qDF.shape', qDF.shape)
    print(' qDF[dt].head(10)')
    print(qDF['dt'].head(10))
    
    print('\nTest 2 - Should NOT return data\n')
    print('   Calling addNewRows...')
    qResult, qDF = addNewRows(qDF, True)
    
    print('\nTest 3 - Sleeping for', mySleepSec, 'seconds...')
    import time
    time.sleep(mySleepSec)
    print('\n         Should return data if enough sleep time.\n')
    qResult, qDF = addNewRows(qDF, True)
    print('\nDone.')


# In[20]:


def checkpoint(lastCheckpoint_in, coreDF_in, verbose_in):
    
    from datetime import datetime
    import pandas as pd
        
    theLastCheckpoint = lastCheckpoint_in
    
    diff     = datetime.now() - lastCheckpoint_in
    diff_sec = int(diff.days*24*60*60 + diff.seconds)
    
    if diff_sec > int(parCore['data_restart_checkpoint_s']):
        
        import time
        t0 = time.perf_counter()
        
        fn = baseDir + parCore['data_restart_csv_file']
        
        print('\nCheckpoint begin', end='...')
        alertLogWrite('Checkpoint begin, ' + str(fn))
        
        try:
            coreDF_in.to_csv (fn, index = False, header=True)
        except:
            line = ' Checkpoint, unable to checkpoint.\n'
            print(line)
            alertLogWrite(line)
        else:
            theLastCheckpoint = datetime.now()

            rows = str(coreDF_in.shape[0])
            et   = str(round(time.perf_counter() - t0,1))
            print('complete.', rows, et + 's', end='')
            alertLogWrite('Checkpoint end, ' + rows + ' rows, ' + et + ' sec')
    
    else:
        if verbose_in:
            print('Checkpoint occured', diff_sec, 'seconds ago.')
    
    return(theLastCheckpoint)

"""
# This test will RESET the checkpoint file. That is, DESTROY the existing checkpoint file

print('Test 1 - Checkpoint should occur')
from datetime import datetime, timedelta
whileBack = datetime.now() + timedelta(days=-1)
last = checkpoint(whileBack, coreDF, True)

print('\nTest 2 - Checkpoint should NOT occur')
checkpoint(last, coreDF, True)    
""" 


# In[21]:


def defineCoreDF(verbose_in):
    try:
        del coreDF
    except:
        pass
    
    myDF = pd.DataFrame(columns=['dt','metric_name', 'metric_value'])
    myDF = resetCoreDFtypes(myDF, verbose_in)
    
    return(myDF)


# In[22]:


# Denormalize for sysmetric type data, with columns dt (yyyymmddhhss), metric_name, metric_value

def denormalize(df_in,verbose_in):
    
    if verbose_in:
        print("Denormalizing")
        print("   BEFORE ", df_in.shape)
    
    # The "values" will become the new "columns" value
    df_inPiv = df_in.pivot_table(index='dt', values='metric_value', columns=['metric_name'])
    df_inPiv.reset_index(inplace=True)

    if verbose_in:
        print("   AFTER  ", df_inPiv.shape)
        print("done.")

    return(df_inPiv)
    
    
if Testing:
    print("Testing Function: denormalize \n")
    
    x = 'ad_test_data_1b.csv'
    x = 'quick_restart-fun.csv'
    result, testDF = loadData(x, True)
    if not result:
        x = 'http://filebank.orapub.com/DataSets/ad_test_data_1b.csv'
        result, testDF = loadData(x, True)
    
    print('\nBEFORE testDF.shape  ', testDF.shape)
    if result:
        print('testDF.columns.list', testDF.columns)
        #features = ['dt','metric_name','metric_value']
        #testDF = testDF[features]
        testDFpiv = denormalize(testDF, True)
        print('\nAFTER testDFpiv.shape', testDFpiv.shape)
        print('testDFpiv.columns.list', testDFpiv.columns)
        print('\ntestDF["dt"].head()')
        print(testDF['dt'].head())

    print('\nDone.')


# In[23]:


def EngineerFeatures(DF_in, verbose_in):
    
    # Expecting DF_in to be denormalized from some v$sysmetric type view
    # Each new feature should have a corresponding T/F entry in the configuration file
    # Using pd.get_dummies to one-hot-encode the new feature
    
    import datetime
    import pandas as pd
    
    if verbose_in:
        print('Engineering features', end=' ... ')
    
    myWorkDF       = DF_in.copy()
    myWorkDF['dt'] = myWorkDF['dt'].astype('int64') 
    
    myWorkDF['datetime'] = pd.to_datetime(myWorkDF['dt'], format='%Y%m%d%H%M%S')
    
    # Hour Of Day - feature_enable_hourofday
    
    if str2bool(parCore['feature_enable_hourofday']):
        if verbose_in:
            print('hourofday', end=' ... ')
        myWorkDF['hourofday']   = myWorkDF['datetime'].dt.hour
        myWorkDF.fillna({'hourofday':0}, inplace = True)
        myWorkDF = pd.concat([myWorkDF, pd.get_dummies(myWorkDF['hourofday'], prefix='hod', dummy_na=True)], axis=1)
        myWorkDF = myWorkDF.drop(columns=['hourofday'])
    
    # Day Of Week - feature_enable_dayofweek
    
    if str2bool(parCore['feature_enable_dayofweek']):
        if verbose_in:
            print('dayofweek', end=' ... ')
        myWorkDF['dayofweek'] = myWorkDF['datetime'].dt.dayofweek
        myWorkDF.fillna({'dayofweek':0}, inplace = True)
        myWorkDF = pd.concat([myWorkDF, pd.get_dummies(myWorkDF['dayofweek'], prefix='dow', dummy_na=True)], axis=1)
        myWorkDF = myWorkDF.drop(columns=['dayofweek'])
    
    # Week Of Month - feature_enable_weekofmonth
    
    if str2bool(parCore['feature_enable_weekofmonth']):
        if verbose_in:
            print('weekofmonth', end=' ... ')
        myWorkDF['weekofmonth'] = myWorkDF['datetime'].apply(lambda d: (d.day-1) // 7 + 1)
        myWorkDF.fillna({'weekofmonth':0}, inplace = True)
        myWorkDF = pd.concat([myWorkDF, pd.get_dummies(myWorkDF['weekofmonth'], prefix='wom', dummy_na=True)], axis=1)
        myWorkDF = myWorkDF.drop(columns=['weekofmonth'])
    
    myWorkDF = myWorkDF.drop(columns=['datetime'])
    
    #
    # Drop unwanted features here... Feature removal
    #
    
    if verbose_in:
        print(myWorkDF.columns)
        print("done.")
    
    return (myWorkDF)

if Testing:
    print('Testing Function: EngineerFeatures')
    parCore = readParamFile(True) # keep result as parCore, so it can be used when testing
    print('     BEFORE', testDFpiv.shape)
    testDFpiv2 = EngineerFeatures(testDFpiv, True)
    print('     AFTER ', testDFpiv2.shape)
    print()
    print(testDFpiv2.columns)
    try:
        colsToShow = ['dt', 'wom_3.0','dow_2.0', 'hod_21.0']
        print(testDFpiv2[colsToShow].tail(10))
        print(testDFpiv2[colsToShow].head(10))
    except:
        pass


# In[24]:


# Function: Dimension Reduction, using either:
#           PCA: Principle Component Analysis
#           ICA: Independent Component Analysis

def DimReduce(df_in, model_in, dimensions_in, verbose_in):
    
    if verbose_in:
        print('Reducing dimensionality using', str(model_in),               'from', str(df_in.shape), 'to', end="... ")
    
    in_shape = df_in.shape
    
    if model_in == 'PCA':
        
        from sklearn.decomposition import PCA                 # load library
        pca       = PCA(n_components=dimensions_in)           # init model
        array_out = pca.fit_transform(df_in)                  # fit DF_in and transform DF_in
        
    elif model_in == 'ICA':
        
        from sklearn.decomposition import FastICA                        # load library
        ICA       = FastICA(n_components=dimensions_in, random_state=12) # init model
        array_out = ICA.fit_transform(df_in)                             # fit DF_in and transform DF_in
        
    else:
        print('   ERROR Function DimReduce. Invalid model provided.')
    
    df_out = pd.DataFrame(data = array_out) # create DF from array
    
    if verbose_in:
        print(str(df_out.shape), 'done.')
        
    return(df_out, array_out)

if Testing:
    print('Testing Function: DimRed2')
    print('     BEFORE', testDFpiv2.shape)
    testDFpiv3, bogus = DimReduce(testDFpiv2.drop(columns=['dt']), 'PCA', 2, True)
    print('     AFTER ', testDFpiv3.shape)
    print()
    print(testDFpiv3.head(4))


# In[25]:


# Function: Scale
#             Standardize: mean=0 stdev=1
#             Normanize  : min =  max  =1

def Scale(df_in, type_in, verbose_in):
    
    if verbose_in:
        print('Scaling', end='... ')
    
    if type_in == 'standardize':
        if verbose_in:
            print('standardizing (mean=0 stdev=1)', end='... ')
        
        from sklearn.preprocessing import StandardScaler      # load lib
        myModel = StandardScaler(with_mean=True, with_std=True).fit(df_in) # init and fit model 
        myAR    = myModel.transform(df_in)                    # scale/transform df_in, result is an array
        myDF    = pd.DataFrame(myAR, columns=df_in.columns)   # convert result array to DF
    
    elif type_in == 'normalize':
        if verbose_in:
            print('normalizing (min=0 max=1)', end='... ')
        
        from sklearn.preprocessing import MinMaxScaler      # load lib
        myModel = MinMaxScaler().fit(df_in)                 # init and fit model 
        myAR    = myModel.transform(df_in)                  # transform/normalize df_in, resuilt is an array
        myDF    = pd.DataFrame(myAR, columns=df_in.columns) # convert result array to DF
    else:
        print('Error in Scale function. type_in=', type_in)
        myDF = df_in
        myAR = df_in.to_numpy()
    
    if verbose_in:
        print("done.")
        
    return(myDF)                                            # return the DF, not the Array


if Testing:
    print('Testing Function: Scale\n')
    print('Raw Data...\n')
    print(testDFpiv2[['Average Active Sessions','Database CPU Time Ratio']].describe())
    print()

    qaDF = Scale(testDFpiv2.drop(columns=['dt']), 'standardize', True)
    print()
    print(qaDF[['Average Active Sessions','Database CPU Time Ratio']].describe())
    
    print()
    qaDF = Scale(testDFpiv2.drop(columns=['dt']), 'normalize', True)
    print()
    print(qaDF[['Average Active Sessions','Database CPU Time Ratio']].describe())
    


# In[26]:


# Function: From a dataframe, create cluster,
# returning the initialized model, fitted model and fitted predict model

def create_cluster(df_in, cluster_type_in, cluster_no_in, verbose_in):
    
    if verbose_in:
        print('Creating', cluster_type_in, 'with', str(cluster_no_in), 'clusters', end='... ')
    
    if cluster_type_in == 'kmeans':
        
        from sklearn.cluster import KMeans
        mymodelinit      = KMeans(n_clusters=cluster_no_in, init='k-means++', max_iter=300, n_init=10, random_state=0)
        mymodelfit       = mymodelinit.fit(df_in)
        mymodelfitlabels = mymodelfit.labels_
        mymodelfitpred   = mymodelfit.predict(df_in)
        
    elif cluster_type_in == 'MiniBatchKMeans':
        
        from sklearn.cluster import MiniBatchKMeans
        mymodelinit      = MiniBatchKMeans(n_clusters=cluster_no_in, batch_size=100, init='k-means++', max_iter=300, n_init=10, random_state=0)
        mymodelfit       = mymodelinit.fit(df_in)
        mymodelfitlabels = mymodelfit.labels_
        mymodelfitpred   = mymodelfit.predict(df_in)
    
    elif cluster_type_in == 'IsolationForest':
        
        my_n_estimators = int(parCore['model_strategy_p1'])
        my_n_jobs       = int(parCore['model_strategy_p2'])
        
        from sklearn.ensemble import IsolationForest
        mymodelinit      = IsolationForest(n_estimators=my_n_estimators, max_samples="auto",                                            contamination="auto", random_state=1, n_jobs=my_n_jobs, verbose=0)
        mymodelfit       = mymodelinit.fit(df_in)
        mymodelfitpred   = mymodelfit.predict(df_in)
    
    else:
        print('ERROR in function, create_cluster')
        
    if verbose_in:
        print('done.')
    
    return(mymodelinit, mymodelfit, mymodelfitpred)


if Testing:
    
    print('Testing Function: create_cluster')
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    parCore = readParamFile(True) # keep result as parCore, so it can be used when testing
    
    from collections import Counter
    from numpy import unique
    
    print('Testing Isolation Forest...\n')
    
    fn = 'quick_restart-fun.csv'
    result, testDF = loadData(x, True)
    if not result:
        result, testDF = loadData(parCore['data_restart_http_base'] + '/' + fn, True)
        testDF.to_csv (fn, index = False, header=True)
        print('Saved', fn)
    
    ModelScaling = 'standardize' # try: standardize, normalize
    testDFpiv = denormalize(testDF, True)
    qaDF = Scale(testDFpiv.drop(columns=['dt']), ModelScaling, True)
    print()
    print(qaDF[['Average Active Sessions','Database CPU Time Ratio']].describe())
    
    plotPointsDF, plotPointsAR = DimReduce(qaDF, 'PCA', 2, True) # try: PCA and ICA. Only affects plotting
    
    print('\nShould see the DimReduce result (scaling has already taken place)')
    print(plotPointsDF.describe())
    
    fig = plt.figure()
    plt.title('Isolation Forest - Raw Points After Preprocessing\n' + 'Scaling Into Model & Plot: ' + str(ModelScaling) )
    plt.scatter(plotPointsAR[:,0], plotPointsAR[:,1], s=1, c='blue')
    plt.show()
    
    myclustertype = 'IsolationForest'
    myclusterno   = -1
    myModelInit, myModelFit, myModelFitPred = create_cluster(qaDF, myclustertype, myclusterno, True)
    print('plotPointsDF, qaDF, myModelFitPred', plotPointsDF.shape, qaDF.shape, myModelFitPred.shape, type(myModelFitPred))
    print('\nDescribing Prediction Results (1 or -1)')
    print(pd.DataFrame(myModelFitPred).describe())
    print(Counter(myModelFitPred))
    print('\nScore Samples:\n')
    print(myModelFit.score_samples(qaDF)) # qaDF has be pivited and scaled, but not Dim Reduced
    print('---------------')
    print(myModelFitPred[:])
    
    # notice i'm loading the plotPointsDF with fitted, scores from qaDF
    # notice i'm multiplying the score by -1, since lower score = more anomalous... i want it reversed
    plotPointsDF['score'] = -1*myModelFit.score_samples(qaDF)
    print('\nDescribing Prediction Result SCORES')
    print(plotPointsDF['score'].describe())
    fig = plt.figure()
    plt.hist(plotPointsDF['score'])
    plt.show()
    
    myScoreThresholdPCT = 0.95 # INCREASE number to see LESS anoamlies... set the threshold higher
    myScoreThresholdVAL = np.quantile(plotPointsDF['score'], myScoreThresholdPCT)
    print('myScoreThresholdPCT, myScoreThresholdVAL', myScoreThresholdPCT, myScoreThresholdVAL)
    plotPointsDF['pointColor'] = 'blue'
    plotPointsDF.loc[plotPointsDF['score'] > myScoreThresholdVAL, 'pointColor'] = 'red'
    print(Counter(plotPointsDF['pointColor']))
    print(plotPointsDF.head())
    fig = plt.figure()
    plt.title('Isolation Forest - Raw Points After Preprocessing & Scoring'  + '\nScaling Into Model: ' + str(ModelScaling)+ '   Score Threshold:' + str(myScoreThresholdPCT))
    plt.scatter(plotPointsDF[0], plotPointsDF[1], s=1, c=plotPointsDF['pointColor'])
    plt.show()
    
    print('\n\n')
    
    modelList = ['kmeans','MiniBatchKMeans']
    
    for myclusterno in range(1,3):
        
        for myclustertype in modelList:

            print()
            qaDF = testDFpiv2.drop(columns=['dt'])
            myModelInit, myModelFit, myModelFitPred = create_cluster(qaDF, myclustertype, myclusterno, True)
            print("")
            print("   Cluster type        :", myclustertype)
            print("   Cluster numbers     :", myclusterno)
            print("   Cluster points      :", len(myModelFitPred))
            print("   Counter Fit Labels  :", Counter(myModelFit.labels_))
            print("   Counter Fit U Labels:", unique(myModelFitPred))
            print("   Cluster model       :", myModelInit)
            print('           center X    :', myModelInit.cluster_centers_[:, 0])
            print('           center Y    :', myModelInit.cluster_centers_[:, 1])
            print('           center Z    :', myModelInit.cluster_centers_[:, 2])
            print('           center Z2   :', myModelInit.cluster_centers_[:, 3])
    
    print("\nDone.")


# In[27]:


# Used when the model produces a score point to centroid distance is not the strategy
# Currently being used with model, Isolation Forest

def getModelScore(DF_in, myModel_in, myModelFit_in, verbose_in):
    
    if verbose_in:
        print('In function, getModelScore. Model:', myModel_in)
    
    DF_in['score']    = -1*myModelFit_in.score_samples(DF_in)
    DF_in['distance'] = DF_in['score'] # distance is used for compatility with other clustering methods
    
    if verbose_in:
        print('\nDescribe')
        print(DF_in[['score','distance']].describe())
        myScoreThresholdVAL = np.quantile(DF_in['score'], float(parCore['model_threshold_value']))
        print('Score Threshold:', parCore['model_threshold_value'], myScoreThresholdVAL)
    
    return(DF_in)

if Testing:
    print('Testing function: getModelScore')
    
    fn = 'quick_restart-fun.csv'
    result, testDF = loadData(x, True)
    if not result:
        result, testDF = loadData(parCore['data_restart_http_base'] + '/' + fn, True)
        testDF.to_csv (fn, index = False, header=True)
        print('Saved', fn)
    
    ModelScaling = 'standardize' # try: standardize, normalize
    testDFpiv = denormalize(testDF, True)
    qaDF = Scale(testDFpiv.drop(columns=['dt']), ModelScaling, True) # try: standardize and normalize
    print()
    print(qaDF[['Average Active Sessions','Database CPU Time Ratio']].describe())
    
    myclustertype = 'IsolationForest'
    myclusterno   = -1
    myModelInit, myModelFit, myModelFitPred = create_cluster(qaDF, myclustertype, myclusterno, True)
    qaDF = getModelScore(qaDF, myclustertype, myModelFit, True)
    
    print('\ngetModelScore function is complete.\n')
    print(qaDF[['score','distance']].describe())


# In[28]:


# Function: Calculate distances between the a given cluster center and
#           every point in the given Dataframe.

def getCentroidDistances(cluster_type_in, cluster_init_in, cluster_fit_in, DF_in, cluster_no_in, verbose_in):
    
    if verbose_in:
        print("Get_point_to_centroid")
    
    # Part 1 - Create a list of distances from the centroid to each point in DF_in
    
    import numpy as np
    from numpy import linalg as LA
    
    mypoints = DF_in.to_numpy() # convert DF to numpy array
    distances=[] # init list
    i = 0
    
    for datapoint in mypoints:
        
        #print(datapoint)        
        distances.append( LA.norm(datapoint - cluster_init_in.cluster_centers_[cluster_no_in]) )        
        i = i + 1
    
    # Part 2 - Add a new feature 'distance' for each point, containing its distance to centroid
    
    DF_out             = DF_in
    DF_out['distance'] = distances  # new feature/column containing their respective point distance
    
    if verbose_in:
        # Calculate statistics
        print('      mean=%0.2f median=%0.2f' % ( np.mean(distances), np.median(distances) ))
        print('      95-pct=%0.2f 98-pct=%0.2f' % ( np.quantile(distances, 0.95), np.quantile(distances, 0.98) ))
        print('      min=%0.2f max=%0.2f' % ( np.min(distances), np.max(distances) ))
        print('done.')
        
    return(DF_out)
    

if Testing:
    print("Testing Function: get_point_to_centroid")
    
    modelList = ['kmeans','MiniBatchKMeans']
    
    for myclustertype in modelList:
    
        print("\n" + myclustertype + str(".......................................\n"))
        
        myclusters    = 1 # number of clusters created
        myclusterNo   = 0 # cluster number to get point centroid details, starting with 0
        qaDF = Scale(testDFpiv2.drop(columns=['dt']), 'standardize', True)
        myModel, myModelFit, myCluster = create_cluster(qaDF, myclustertype, myclusters, True)
        print()
        print("      Cluster type   :", myclustertype)
        print("      Cluster numbers:", myclusters)
        print("      Cluster points :", len(myCluster))
        print("      Cluster model  :", myModel)
        print()
        
        myPointsDF = getCentroidDistances(myclustertype, myModel, myCluster, qaDF, myclusterNo, True)
        
        print()
        print('len(myPointsDF)',len(myPointsDF))
        print('myPointsDF.shape', myPointsDF.shape)
        print(myPointsDF[['Average Active Sessions','distance']].head(4))

    print("\nDone Testing Function: get_point_to_centroid")


# In[29]:


def getChartFN(chartsDir_in, sampleTime_in, anomalyTrue_in):
    
    return( chartsDir_in + '/' + str(sampleTime_in) + '-' + str(anomalyTrue_in) + '.png' )

if Testing:
    print( getChartFN('2Ddir', 12345, True) )


# In[30]:


def preprocessCoreDF(DF_in, verbose_in, showTime_in):
        
    import time
    t0 = time.perf_counter()
    
    if verbose_in:
        print('preprocessing now...', end='')
    
    myClusterNo       = 1
    myClusterCenterNo = 0

    DFpiv                          = denormalize(DF_in, verbose_in)
    DFpivEng                       = EngineerFeatures(DFpiv, verbose_in)
    DFpivEngScale                  = Scale(DFpivEng.drop(columns=['dt']), 'standardize', verbose_in)
    myModel, myModelFit, myCluster = create_cluster(DFpivEngScale, parCore['model_strategy'], myClusterNo, verbose_in)
    if parCore['model_threshold_strategy'] == 'CentroidPercentile':
        DFpivEngScaleDist = getCentroidDistances(parCore['model_strategy'], myModel, myCluster, DFpivEngScale, myClusterCenterNo, verbose_in)
    elif parCore['model_threshold_strategy'] == 'ScorePercentile':
        DFpivEngScaleDist = getModelScore(DFpivEngScale, myModel, myModelFit, verbose_in)
    else:
        print('ERROR: model_threshold_strategy parameter set to invalid value.')
    
    if verbose_in:
        print()
        print('DFpiv, DFpivEngScaleDist', DFpiv.shape, DFpivEngScaleDist.shape )
        print()
        print(DFpivEngScaleDist[['distance','Average Active Sessions']].describe())
        print()
        print(DFpivEngScaleDist.columns)
       
    if showTime_in:
        line = f'PP {time.perf_counter() - t0:0.2f}s'
        print(' (' + line + ')')
        alertLogWrite(line)
    else:
        print()
    
    return(DFpiv, DFpivEng, DFpivEngScale, DFpivEngScaleDist)

if Testing:
    print("Testing Function: preprocessCoreDF")
    QhaveRestartData, QrestartDataDF, QrestartOldestDt, QrestartNewestDt = loadRestartData(True, True)

    if QhaveRestartData:
        print('Restart data loaded from', QrestartOldestDt, 'to', QrestartNewestDt, QrestartDataDF.shape[0], 'rows')
        qDF = QrestartDataDF.copy()
        qDF = resetCoreDFtypes(qDF, True)
        qDFpiv, qDFpivEng, qDFpivEngScale, qDFpivEngScaleDist = preprocessCoreDF(qDF, True, True)
        print('Result: qDFpiv, qDFpivEngScaleDist', qDFpiv.shape, qDFpivEngScaleDist.shape )
        print('        distance median', qDFpivEngScaleDist['distance'].median())
    else:
        print('No restart data available, so not testing')
    


# In[31]:


def alertConditionsMet(DF_in, verbose_in):
    
    myResult = False
    
    if verbose_in:
        print('\nIn alertConditionsMet function', end=' ... ')
    
    fullPass = False
    check1   = False
    check2   = False
    check3   = False
    check4   = False
    
    # Check 1. Alerting must be enabled
    if str2bool(parCore['alert_enable']):
        if verbose_in:
            print('alert_enable',parCore['alert_enable'], end=' ... ' )
        check1 = True
    
    # Check 2. Must have enough sample sets to warrent a legitamet anomaly detection check
    mySamplesSets = DF_in.dt.nunique() 
    
    if mySamplesSets < int(parCore['alert_min_sample_sets']):
        print(' ', mySamplesSets, 'sample sets below threshold of', parCore['alert_min_sample_sets'])
    else:
        check2 = True
    
    # Check 3. Must have sufficient gap since previous alert
    from datetime import datetime
    timeSinceLastAlert = datetime.now() - lastAlert
    secsSinceLastAlert = int(timeSinceLastAlert.days*24*60*60 + timeSinceLastAlert.seconds)
    if verbose_in:
        print(secsSinceLastAlert, 'secs since last alert, need sec gap of', parCore['alert_min_secs_between'])
    
    if secsSinceLastAlert > int(parCore['alert_min_secs_between']):
        check3 = True
    
    # Check 4. Check if Force Alert enabled
    if str2bool(parCore['debug_force_anomaly_enable']):
        print('\ndebug_force_anomaly_enable is TRUE.')
        check4 = True
    
    if (check1 and check2 and check3) or check4 :
        fullPass = True
    
    if fullPass and verbose_in:
        print('yes, conditions met with', mySamplesSets, 'sample sets')
    elif not fullPass and verbose_in:
        print('check1', check1, 'check2', check2, 'check3', check3, 'check4', check4)
    
    return(fullPass)

if Testing:
    parCore = readParamFile(False)
    from datetime import datetime
    lastAlert = datetime.now()
    print(alertConditionsMet(testDF, True))


# In[32]:


def flagAllAnomalies(DFpivEngScaleDist_in, verbose_in):
    
    import numpy as np
    from collections import Counter
    from numpy import unique
    
    if verbose_in:
        print('Flagging all anomalies using', parCore['model_threshold_strategy'], 'strategy', end=' ')
    
    if parCore['model_threshold_strategy'] == 'CentroidPercentile' or        parCore['model_threshold_strategy'] == 'ScorePercentile':
    
        thresholdPct = float(parCore['model_threshold_value'])
    
    else:
        print('ERROR: function: flagAllAnomalies. Invalid parameter value for, model_threshold_strategy')
    
    # For all models...
    threshold    = np.quantile(DFpivEngScaleDist_in['distance'].to_numpy(), thresholdPct)
    DFpivEngScaleDist_in['threshold'] = threshold
    
    DFpivEngScaleDist_in['anomaly'] = False
    DFpivEngScaleDist_in.loc[DFpivEngScaleDist_in['distance'] >= DFpivEngScaleDist_in['threshold'], 'anomaly'] = True
    
    if verbose_in:
        medDist = DFpivEngScaleDist_in['distance'].median().round(3)
        maxDist = DFpivEngScaleDist_in['distance'].max().round(3)
        print(f'TD={threshold:0.3f} medD/maxD={medDist:0.3f}/{maxDist:0.3f}')
        print(Counter(DFpivEngScaleDist_in['anomaly']))
    
    return DFpivEngScaleDist_in

if Testing:
    parCore = readParamFile(False)
    qDFpivEngScaleDistFlagged = flagAllAnomalies(qDFpivEngScaleDist, True)
    print()
    print('qDFpivEngScaleDistFlagged', qDFpivEngScaleDistFlagged.shape)
    print()
    print(qDFpivEngScaleDistFlagged[['distance','threshold','anomaly']].head(50))


# In[33]:


# Function: Chart Plot - anomalous or not to screen and file

# pointsDF_in    - Returned DF from flagAllAnomalies function: includes points, distances, thresholds, anomaly T/F
# dataDatesDF_in - Sample date/time of each row
# checkrowIdx_in - Which row to check for an anomaly. Testing, could be anything, otherwise most recent
# dim_in         - dimensionality of chart: 2 or 3
# actionTime_in  - when the action occured, local client-side time.

def chartPlot(pointsDF_in, dataDatesDF_in, checkRowIdx_in, dim_in:int, actionTime_in, verbose_in):
    
    import matplotlib.pyplot as plt
    #from matplotlib import pyplot as plt
    from datetime import datetime
    #from datetime import date
    
    plotPointsDF, plotPointsAR = DimReduce(pointsDF_in.drop(columns=['distance','threshold','anomaly']), parCore['dim_reduce_model'], dim_in, verbose_in)
    plotPointsDF['anomaly']    = pointsDF_in['anomaly']
    
    anomalyTrue = pointsDF_in['anomaly'].to_numpy()[checkRowIdx_in]
    myDistance  = pointsDF_in['distance'].to_numpy()[checkRowIdx_in]
    myThreshold = pointsDF_in['threshold'].to_numpy()[checkRowIdx_in]
    
    plotPointsDF['pointColor'] = 'blue'
    plotPointsDF.loc[plotPointsDF['anomaly'] == True, 'pointColor'] = 'magenta'
    if verbose_in:
        print(Counter(plotPointsDF['pointColor']))
        print(plotPointsDF.head())
        
    fig_size    = plt.rcParams["figure.figsize"]
    fig_size[0] = float(parCore['alert_chart_width_inch'])
    fig_size[1] = float(parCore['alert_chart_height_inch'])
    plt.rcParams["figure.figsize"] = fig_size
    
    
    if dim_in == 2:
        
        """
        myScoreThresholdPCT = 0.95 # INCREASE number to see LESS anoamlies... set the threshold higher
        myScoreThresholdVAL = np.quantile(plotPointsDF['score'], myScoreThresholdPCT)
        print('myScoreThresholdPCT, myScoreThresholdVAL', myScoreThresholdPCT, myScoreThresholdVAL)
        
        plotPointsDF['pointColor'] = 'blue'
        plotPointsDF.loc[plotPointsDF['score'] > myScoreThresholdVAL, 'pointColor'] = 'red'
        print(Counter(plotPointsDF['pointColor']))
        print(plotPointsDF.head())
        
        fig = plt.figure()
        plt.title('Isolation Forest - Raw Points After Preprocessing & Scoring'  + '\nScaling Into Model: ' + str(ModelScaling)+ '   Score Threshold:' + str(myScoreThresholdPCT))
        plt.scatter(plotPointsDF[0], plotPointsDF[1], s=1, c=plotPointsDF['pointColor'])
        plt.show()
        """
        
        # Initialize 2D plot
        fig = plt.figure()
        
        # Plot all points
        plt.scatter(plotPointsDF[0], plotPointsDF[1], s=int(parCore['alert_chart_point_size_normal']), c=plotPointsDF['pointColor'])
        #plt.scatter(plotPointsAR[:,0], plotPointsAR[:,1], s=5, c='blue')
        
        # Plot the point to check (normally the most recent point is what we are checking)
        if anomalyTrue:
            plt.scatter(plotPointsAR[checkRowIdx_in,0], plotPointsAR[checkRowIdx_in,1], s=int(parCore['alert_chart_point_size_anom']), c='red')
        else:
            plt.scatter(plotPointsAR[checkRowIdx_in,0], plotPointsAR[checkRowIdx_in,1], s=int(parCore['alert_chart_point_size_anom']), c='green')
    
    elif dim_in == 3:
        # https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html
        #
        # Initialize 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(float(parCore['alert_chart_3D_elevate_angle']), float(parCore['alert_chart_3D_horizontal_angle']))
        
        # Plot all points
        ax.scatter(plotPointsAR[:,0], plotPointsAR[:,1], plotPointsAR[:,2], s=int(parCore['alert_chart_point_size_normal']), c=plotPointsDF['pointColor'], marker='o')
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        ax.set_zlabel(' ')
        
        # Plot the point to check
        if anomalyTrue:
            ax.scatter(plotPointsAR[checkRowIdx_in,0], plotPointsAR[checkRowIdx_in,1], plotPointsAR[checkRowIdx_in,2], s=int(parCore['alert_chart_point_size_anom']), c='red')
        else:
            ax.scatter(plotPointsAR[checkRowIdx_in,0], plotPointsAR[checkRowIdx_in,1], plotPointsAR[checkRowIdx_in,2], s=int(parCore['alert_chart_point_size_anom']), c='green')
    else:
        print('Error in chartPlot function. Invalid dim_in value,', dim_in)
    
    if anomalyTrue:
        mytitle1 = 'ANOMALY DETECTED'
    else:
        mytitle1 = 'Anomaly NOT Detected'
    
    # https://strftime.org
    # date/time of sample based on database
    intDT      = int(dataDatesDF_in['dt'].tail(1)) 
    sampleDT   = datetime.strptime(str(intDT), '%Y%m%d%H%M%S')
    databaseDT = sampleDT.strftime('%d-%b-%Y %H:%M:%S')
    
    # single time, used for both chart title and chart png filename
    theTimeNow = datetime.now()
    
    # chart also displays local time
    localDT    = theTimeNow.strftime('%d-%b-%Y %H:%M:%S')
    
    mytitle2 = 'DB: {dt}   Local: {lt} '.format(dt=databaseDT, lt=localDT)
    mytitle3 = 'Distance/Score={dist:0.3f} Threshold={thresh:0.3f}'.format(dist=myDistance, thresh=myThreshold)
    mytitle4 = 'Model:' + parCore['model_strategy'] + ' Threshold Model:' + parCore['model_threshold_strategy'] + ' Value:' + parCore['model_threshold_value']
    
    plt.title(mytitle1 + '\n' + mytitle2 + '\n' + mytitle3 + '\n' + mytitle4)
    
    if dim_in == 2:
        chartFN = getChartFN(chartsDir2D, actionTime_in, anomalyTrue)
    elif dim_in == 3:
        chartFN = getChartFN(chartsDir3D, actionTime_in, anomalyTrue)
    else:
        print('Function chartPlot. Invalid dim_in', dim_in)
    
    plt.savefig(chartFN)
    
    if InNotebook:
        plt.show()
    
    return(chartFN)

if Testing:
    parCore = readParamFile(False)
    
    print("Testing Function: chart_anom")
    print(qDFpiv['dt'].head())
    
    testrowidx = 5
    chartPlot(qDFpivEngScaleDistFlagged, qDFpiv, testrowidx, 2, '20210309151742', True)
    chartPlot(qDFpivEngScaleDistFlagged, qDFpiv, testrowidx, 3, '20210309151742', False)
    print("Done Testing Function: chart_anom")


# In[34]:


def sendAnomalyDetectedEmail(chart2dFN_in, chart3dFN_in, verbose_in):
    
    # The actual email SEND functions are near the top of this file
    
    # Ensure your AnomDetect_Email.py and AnomDetect_DB.py parameters are set correctly.
    import os
    import AnomDetect_DB as db
    import AnomDetect_Email as email
    import importlib

    importlib.reload(db)
    importlib.reload(email)
    
    myResult = False
    
    subject  = 'Anomaly detected in system ' + str(db.dsn)
    message3 = 'There has been anomaly detected in system ' + str(db.dsn) + ' at ' + datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    message4 = '2D chart filename: ' + str(chart2dFN_in)
    message5 = '3D chart filename: ' + str(chart3dFN_in)
    message2 = 'Anomaly Detection System: ' + str(lastCodeUpdate)
    message1 = 'From: ' + str(email.sender_email)
    
    message  = str(message1 + '\n' + message2 + '\n\n' + message3 + '\n' + '\n' + message4 + '\n' + message5)
    
    myLine   = 'Sending alert email from ' + str(email.sender_email) + ' to ' + str(email.receiver_email)
    
    print(myLine, end='...')
    
    if verbose_in:
        print('\nMessage is:')
        print(message, '\n')
    
    if not verbose_in:
        if str2bool(parCore['alert_email_wo_charts']):       
            myResult = sendAlertEmail_wo_charts(subject, message, True)
        
        if str2bool(parCore['alert_email_w_charts']):
            myFN = os.path.basename(chart2dFN_in)
            myResult = sendAlertEmail_w_charts(subject, message, myFN, True)
    
    if myResult:
        myLineEnd = ' was successfully sent.'
    else:
        myLineEnd = ' was NOT successfully sent.'

    print(myLineEnd)
    alertLogWrite(myLine + myLineEnd)
    
if Testing:
    print('Testing sendTextEmail')
    # verbose=True will NOT send message, but will show message text
    sendAnomalyDetectedEmail('Tester 2D filename', 'Tester 3D filename', True)

print('Done.')
        


# In[35]:


def anomalyAction(fullDF_in, pivDF_in, verbose_in):
    
    from datetime import datetime
    
    myDebug = False
    
    # This is THE time of the action. It is used for many things, like: chart filename, header, local time
    actionTime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if verbose_in:
        print('anomalyAction function inputs:')
        print('fullDF_in, pivDF_in, verbose_in')
        print(fullDF_in.shape, pivDF_in.shape, verbose_in)
    
    forceAnomalyTrue = str2bool(parCore['debug_force_anomaly_enable'])
    
    mostRecentRow    = fullDF_in.shape[0]-1
    isAnomalyTrue    = forceAnomalyTrue or fullDF_in['anomaly'].to_numpy()[mostRecentRow]
    myDistance       = fullDF_in['distance'].to_numpy()[mostRecentRow]
    myThreshold      = fullDF_in['threshold'].to_numpy()[mostRecentRow]
    
    medDist          = fullDF_in['distance'].median()
    maxDist          = fullDF_in['distance'].max()
    
    modStrat = parCore['model_strategy']
    
    stats = f'RID={mostRecentRow} D={myDistance:0.2f} T={myThreshold:0.2f} {isAnomalyTrue} medD/maxD={medDist:0.2f}/{maxDist:0.2f} {modStrat}'
    
    if str2bool(parCore['alert_console']) or verbose_in:
        line = 'Anomaly '
        if isAnomalyTrue != True:
            line = line + 'NOT detected'
        else:
            line = line + 'DETECTED'
        if forceAnomalyTrue:
            line = line + ' but FORCED'
    
        print(line, stats)

    if str2bool(parCore['alert_log_enable']) or verbose_in:
        alertLogWrite(line + ', ' + stats)
        
    # alert_chart must be True or no emails will be sent
    
    if str2bool(parCore['model_chart_display_always']) or        (isAnomalyTrue and str2bool(parCore['alert_chart'])) or        verbose_in:
        
        chart2dFN = chartPlot(fullDF_in, pivDF_in, mostRecentRow, 2, actionTime, verbose_in)
        chart3dFN = chartPlot(fullDF_in, pivDF_in, mostRecentRow, 3, actionTime, verbose_in)
        
        if (str2bool(parCore['alert_email_w_charts']) or str2bool(parCore['alert_email_wo_charts'])):
            
            sendAnomalyDetectedEmail(chart2dFN, chart3dFN, verbose_in)
            
        else:
            if verbose_in:
                print('Not sending an alert email.')
        
    if (isAnomalyTrue and str2bool(parCore['alert_log_enable'])) or myDebug:
        
        myFN2D = getChartFN(chartsDir2D, actionTime, isAnomalyTrue)
        myFN3D = getChartFN(chartsDir3D, actionTime, isAnomalyTrue)
        
        myLine = 'Alert triggered, sample=, ' + str(mostRecentRow) +               ', distance=, ' + str(round(myDistance,2)) + ', threshold=, ' + str(round(myThreshold,2)) +               ', 2D=, ' + str(myFN2D) + ', 3D=, ' + str(myFN3D) + '\n'
        
        alertLogWrite(myLine)
        
        if verbose_in:
            print('\n\n' + myLine + '\n\n')
    
    return(isAnomalyTrue) # return isAnomalyTrue is needed to set the lastAlert

if Testing:
    print("Testing Function: anomalyAction")
    print('qDFpivEngScaleDistFlagged', qDFpivEngScaleDistFlagged.shape, 'qDFpiv', qDFpiv.shape)
    anomalyAction(qDFpivEngScaleDistFlagged, qDFpiv, True)

print('Done')
 


# In[36]:


def doCommandLineChecks():
    
    if InNotebook:
        if CmdCheck == 'db':
            print('\nConnection test result:', checkConnectAndQuery(False))
            print('\nYou may seen an exception message below, that can be ignored.')
            doExit()
        elif CmdCheck == 'email':
            print('\nEmail test result:', sendAlertEmail_wo_charts('Test subject line', 'This is a test message', False))
            print('Check your inbox.')
            print('\nYou may seen an exception message below, that can be ignored.')
            doExit()
    else:
        if len(sys.argv) > 1 and str(sys.argv[1]) == 'db':
            print('\nConnection test result:', checkConnectAndQuery(False))
            doExit()
        elif len(sys.argv) > 1 and str(sys.argv[1]) == 'email':
            print('\nEmail test result:', sendAlertEmail_wo_charts('Test subject line', 'This is a test message', False))
            print('Check your inbox.')
            doExit()

if Testing:
    doCommandLineChecks()

print('Done.')


# # Main Program

# In[37]:


import AnomDetect_DB as db

import sys
import pandas as pd
import time
from datetime import datetime, timedelta

line = str('Starting Main Program at ' + datetime.now().strftime("%d-%b-%Y %H:%M:%S") + ' Last update ' + lastCodeUpdate)
print(line, '\n')
alertLogWrite(line)

parCore = readParamFile(True)
print()

doCommandLineChecks()

# set global verbose variable that any function can use
mainVerbose = str2bool(parCore['debug_detail_enable'])

# Email Alerting: Ensure your AnomDetect_Email.py parameters are set correctly.
import AnomDetect_Email as email
line = parCore['alert_email_w_charts'] + ', alert emails with CHARTS are enabled'
print(line)
alertLogWrite(line)
line = parCore['alert_email_wo_charts'] + ', alert emails with NO CHARTS are enabled'
print(line)
alertLogWrite(line)
line = 'Emails are sent from ' + str(email.sender_email) + ' to ' + str(email.receiver_email)
print(line, '\n')
alertLogWrite(line)

# create our inital core dataframe
coreDF = defineCoreDF(mainVerbose)
rawDF  = defineCoreDF(mainVerbose)

# Perform basic DB and Query check before entering main loop
check1 = checkConnectAndQuery(mainVerbose)
check2, dbConnection = makeDbConnection(show_error_in=True,verbose_in=True)

if check1 and check2:
    haveRestartData, restartDataDF, restartOldestDt, restartNewestDt = loadRestartData(mainVerbose, True)

    if haveRestartData:
        print('Restart data loaded from', restartOldestDt, 'to', restartNewestDt, restartDataDF.shape[0], 'rows')
        rawDF = restartDataDF.copy()
        rawDF = resetCoreDFtypes(rawDF, mainVerbose)
    else:
        print('Restart data not available.')
    
    # Alerting can be disabled, while data is still being collected.
    print('Alerting is enabled,', parCore['alert_enable'], 'and minimum sample sets at', parCore['alert_min_sample_sets'])
    
    print('Starting collection at ' + datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
    
    lastAlert      = datetime.today() - timedelta(days=1)
    lastCheckpoint = datetime.now()
    
    # THE MAIN LOOP
    
    while True:
        mainVerbose = str2bool(parCore['debug_detail_enable'])
        
        print('\n' + datetime.now().strftime("%d-%b %H:%M:%S"), end=' ' )
        samplesBefore = rawDF.shape[0]
        
        newSamplesLoaded, rawDF = addNewRows(rawDF, mainVerbose)
        
        if newSamplesLoaded:
            
            #print('\n1 coreDF.shape, rawDF.shape',coreDF.shape, rawDF.shape)
            coreDF = rawDF.tail(int(parCore['data_max_samples_modeled'])).copy()
            coreDF = resetCoreDFtypes(coreDF, mainVerbose)
            #print('2', coreDF['dt'].min(), coreDF['dt'].max())
            coreDF = coreDF[coreDF['dt'] > coreDF['dt'].min()].copy()
                    
            #print('3 coreDF.shape, rawDF.shape',coreDF.shape, rawDF.shape)
            #print('4', coreDF['dt'].min(), coreDF['dt'].max())
            
            samplesAfter = rawDF.shape[0]
            
            if (samplesAfter > samplesBefore) and alertConditionsMet(coreDF, mainVerbose):
                print(' coreDF', coreDF.shape, end='')
                if str2bool(parCore['data_max_samples_show_details']):
                    print('\n                coreDF.dt oldest/newest ' + str(coreDF['dt'].min()) +                           '/' + str(coreDF['dt'].max()), end='' )
                
                DFpiv, DFpivEng, DFpivEngScale, DFpivEngScaleDist = preprocessCoreDF(coreDF, mainVerbose, True)
                DFpivEngScaleDistFlagged                          = flagAllAnomalies(DFpivEngScaleDist, mainVerbose)
                if anomalyAction( DFpivEngScaleDistFlagged, DFpiv, mainVerbose ):
                    lastAlert = datetime.today()
        
        if checkDBFileChange():
                check2, dbConnection = makeDbConnection(show_error_in=True,verbose_in=True)
        
        time.sleep(int(parCore['sample_frequency_sec']))
        
        parCore        = readParamFile(False)
        lastCheckpoint = checkpoint(lastCheckpoint, rawDF, False)
    
else:
    doExit()


# In[ ]:





# In[ ]:




