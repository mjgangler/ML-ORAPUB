#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Machine Learning Data Collector")

lastUpdate = "17-Apr-2021"
print("Last update:", lastUpdate)

TESTING     = False
IN_NOTEBOOK = False
CMD_CHECK   = '' # Only used if InNotebook, then normally, set to '' or to do a check, set to either 'db' or 'email'

print("\nDone.")


# In[ ]:


print("Loading core Python libraries", end="...")

import numpy  as np # for number crunching
import pandas as pd # for dataframe manipulation and preprocessing
import os
from collections import Counter
import cx_Oracle # notice the uppercase O in Oracle... different from the package name
from datetime import datetime, timedelta
import time

print("done.")


# In[ ]:


# Key Settings

import sys
import os

#PARE_CORE = ""

BASE_DIR    = os.getcwd() + "/"

CHARTS_DIR_2D = BASE_DIR + "charts2D"   # <-------- Make sure this directory exists!!!
CHARTS_DIR_3D = BASE_DIR + "charts3D"   # <-------- Make sure this directory exists!!!
ALERT_FN      = BASE_DIR + "ml_dc_alertlog.txt"  # Not really used at the moment
CONFIG_FILE   = BASE_DIR + "ML_DC.cfg"

print()
print("Directories and files:")
print("  BASE_DIR     ", BASE_DIR)
print("  CHARTS_DIR_2D", CHARTS_DIR_2D)
print("  CHARTS_DIR_3D", CHARTS_DIR_3D)
print("  CONFIG_FILE  ", CONFIG_FILE)
print("  ALERT_FN     ", ALERT_FN)

print("\nDone.")


# In[ ]:


if TESTING:
    G_current_seq_num = 0
    G_sysmetric_last_beg_time = datetime.now() - timedelta(days=1)
    G_sysmetric_last_end_time = datetime.now() - timedelta(days=1)


# In[ ]:


def doExit():
    import sys
    print('Exiting clean')
    try:    
        cursor.close()
        dbConnection.close()
    except:
        pass
    
    sys.exit() 
    
print("\nDone.")


# In[ ]:


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

print("\nDone.")


# In[ ]:


def alertLogWrite(line_in):

    from datetime import datetime

    now = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    f = open(ALERT_FN,"a")
    f.write(now + ", " + line_in + "\n")
    
if TESTING:
    x = "Yo Craig, This is an alert log test."
    alertLogWrite(x)

print("\nDone.")


# In[ ]:


def checkDBFileChange():

    # Check if DB access credentials have changed in the ML-DC-DB.py file.
    
    change = False
    
    import ML_DC_DB as db
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

if TESTING:
    print('Testing Function: checkDBFileChange')
    print(checkDBFileChange())

print("\nDone.")


# In[ ]:


def readParamFile(verbose_in):
    
    # ref: https://zetcode.com/python/configparser
    
    import configparser
    
    config = configparser.ConfigParser()
    
    fullConfigFile = CONFIG_FILE
    config.read(fullConfigFile)
    
    try:
        prior = G_par_core
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

if TESTING:
    print('Testing Function: readParamFile')
    # pareCore is outside of the function, so results available everywhere
    G_par_core = readParamFile(True) # keep result as parCore, so it can be used when testing
    


# In[ ]:


def closeDbConnection(verbose_in):
    
    import cx_Oracle
    
    if verbose_in:
        print("Closing DB connection", end="...")
    try:
        cursor.close()
        dbConnection.close()
    except:
        pass
    
    if verbose_in:
        print("completed.")

if TESTING:
    closeDbConnection(True)

print("\nDone.")


# In[ ]:


def openDbConnection(show_error_in, verbose_in):
    
    import ML_DC_DB as db
    import cx_Oracle
    import os
    import importlib
    
    
    importlib.reload(db)
    
    conTF = False
    
    if verbose_in:
        print('Attempting Oracle connection db.user db.dsn ', db.user, db.dsn, end=' ...')
    
    try:
        closeDbConnection(verbose_in)
    except:
        pass
    
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
            print('Connection success.')
            print('Connected Oracle version', conDetails.version)
        
    return( conTF, conDetails )


if TESTING:
    print('Testing Function: makeDbConnection \n')
    
    print('Test 1')
    result, conQ = openDbConnection(show_error_in=True,verbose_in=True)
    print('Test 1 result', result, 'dbConnection', conQ)
    
    print('\nTest 2')
    result, conQ = openDbConnection(show_error_in=True,verbose_in=False)
    print('Test 2 result', result, 'dbConnection', conQ)
 
    print('\nDone.')


# In[ ]:


def runSQL(sql_in, verbose_in):
    
    if verbose_in:
        print("\nENTER Function runSQL")
    
    myResult = False
    
    DF = pd.DataFrame()
    
    if verbose_in:
        print("sql:", sql_in)
    
    try:
        result, myDbConnection = openDbConnection(show_error_in=True, verbose_in=verbose_in)
        
        DF = pd.read_sql(sql_in, con=myDbConnection)
        
        if verbose_in:
            print("Result test shape", DF.shape)
        
        myResult = True
    
    except:
        print("Error, runSQL, failed:", sql_in)
    
    try:
        closeDbConnection(verbose_in)
    except:
        pass
    
    if verbose_in:
        print("\nEXIT Function runSQL")
        
    return myResult, DF

if TESTING:
    qTF, qDF = runSQL("select * from dual", True)
    print(qTF)
    print(qDF)


# In[ ]:


def querySourceCheck(object_in, verbose_in):
            
    import cx_Oracle
    
    myResult = False
    
    if verbose_in:
        print('Attempting to query from', object_in)
    
    parObjectName  = object_in + "_object_name"
    parWhereClause = object_in + "_where_clause"
    
    sql = "select count(*) from " + G_par_core[parObjectName]     + " where " + G_par_core[parWhereClause] + " and rownum < 5"
    
    myResult, DF = runSQL(sql, verbose_in)
    
    return(myResult, DF)


if TESTING:
    print('Testing Function: querySourceCheck')
    
    print("\nTest 1 ASH verbose:True")
    qTF, qDF = querySourceCheck("ash", True)
    print("Test Result", qTF, qDF.shape)
    
    print("\nTest 2 SYSMETRIC verbose:False")
    qTF, qDF = querySourceCheck("sysmetric", False)
    print("Test Result", qTF, qDF.shape)
    
    print("\nTest 3 LABEL verbose:False")
    qTF, qDF = querySourceCheck("label", False)
    print("Test Result", qTF, qDF.shape)

print("\nDone.")


# In[ ]:


def IsNewDataAvailable(verbose_in):
    
    # Note: Using a SYSMETRIC end_time increase to indicate new data is available.
    
    if verbose_in:
        print("\nENTER Function IsNewDataAvailable")
    
    NewData = False
    begDate = datetime.today() - timedelta(days=100) # just to get a valid date type
    endDate = datetime.today() - timedelta(days=100) # just to get a valid date type
    
    sql = "select min(begin_time) BT, max(end_time) ET from " + G_par_core["sysmetric_object_name"]     + " where " + G_par_core["sysmetric_where_clause"]
    
    runSQLTF, DF = runSQL(sql, verbose_in)
    
    if runSQLTF:
        if DF.ET[0] > G_sysmetric_last_end_time:
            NewData = True
            endDate = DF.ET[0]
            begDate = DF.BT[0]
    
    if verbose_in:
        print("\nEXIT Function IsNewDataAvailable")
    
    return NewData, begDate, endDate

if TESTING:
    qTF, qBegDate, qEndDate = IsNewDataAvailable(True)
    if qTF:
        G_sysmetric_last_beg_time = qBegDate
        G_sysmetric_last_end_time = qEndDate
    print("1", qTF, qBegDate, qEndDate)
    print("2     ", G_sysmetric_last_beg_time, G_sysmetric_last_end_time)
    #print("3     ", G_sysmetric_last_beg_time.strftime("%d-%b-%Y %H:%M:%S"), G_sysmetric_last_end_time.strftime("%d-%b-%Y %H:%M:%S"))
    
    #print("\nSleeping 2s...\n")
    #time.sleep(2)

print("\nDone.")


# In[ ]:


def getSysmetricData(sysmetric_last_end_time_in, verbose_in):
    
    if verbose_in:
        print("\nENTER Function getSysmetricData")
    
    sysmetric_last_end_time = sysmetric_last_end_time_in.strftime("%d-%b-%Y %H:%M:%S")
    
    sql = "select begin_time sample_time_beg, end_time sample_time_end, metric_name, value metric_value from "     + G_par_core["sysmetric_object_name"]     + " where " + G_par_core["sysmetric_where_clause"]     + "   and end_time > to_date('" + sysmetric_last_end_time + "','DD-Mon-YYYY HH24:MI:SS')"     + " order by 1, 2"
    
    runSQLTF, DF = runSQL(sql, verbose_in)
    
    if verbose_in:
        print("\nEXIT Function getSysmetricData")
    
    return runSQLTF, DF

if TESTING:
    qsysTF, qsysDF = getSysmetricData(G_sysmetric_last_end_time, True)
    print(qsysDF.dtypes)
    G_sysmetric_last_beg_time = qsysDF['SAMPLE_TIME_BEG'].min()
    G_sysmetric_last_end_time = qsysDF['SAMPLE_TIME_END'].max()
    print("G_sysmetric_last_beg_time", G_sysmetric_last_beg_time)
    print("G_sysmetric_last_end_time", G_sysmetric_last_end_time)
    print(qsysTF, qsysDF.shape)
    print(qsysDF.head(5))


# In[ ]:


def getAshData(minTime_in, maxTime_in, verbose_in):
    
    minTime = minTime_in.strftime("%d-%b-%Y %H:%M:%S")
    maxTime = maxTime_in.strftime("%d-%b-%Y %H:%M:%S")
    
    if verbose_in:
        print("\nENTER Function getAshData, minTime", minTime, " maxTime", maxTime)
    
    #sql = "select sample_time, sample_id, session_id, session_state, session_type, wait_class, module from " \
    sql = "select sample_id, session_state, session_type, wait_class, module, sql_opcode, pga_allocated, temp_space_allocated from "     + G_par_core["ash_object_name"]     + " where " + G_par_core["ash_where_clause"]     + "   and sample_time > to_date('" + minTime + "','DD-Mon-YYYY HH24:MI:SS')"     + "   and sample_time < to_date('" + maxTime + "','DD-Mon-YYYY HH24:MI:SS')"     + " order by sample_id, session_id"
        
    runSQLTF, DF = runSQL(sql, verbose_in)
    
    if verbose_in:
        print(DF.head())
    
    if verbose_in:
        print("\nEXIT Function getAshData")
    
    return runSQLTF, DF


# In[ ]:


#if TESTING:
#    
#    qTF, qDF = getAshData(G_sysmetric_last_beg_time, G_sysmetric_last_end_time, True)
#    print(qTF, qDF.shape)
#    print(qDF.head(5))


# In[ ]:


# This is not completed by any means...

def getLabelData(lastSampleDate_in, verbose_in):
    
    sql = "select sample_time, sample_id, session_id, sql_id, module from "     + G_par_core["label_object_name"]     + " where " + G_par_core["label_where_clause"]     + "   and sample_time > to_date('" + lastSampleDate_in + "','DD-Mon-YYYY HH24:MI:SS')"     + " order by sample_id, session_id"
    
    runSQLTF, DF = runSQL(sql, verbose_in)
    
    return runSQLTF, DF

print("\nDone.")


# In[ ]:


def getNewData(sysDF, ashDF, labelDF, verbose_in):
    
    # This function will:
    #    1. Collect the new SYSMETRIC (sysNewDF) and ASH (ashNewDF) and LABEL (labelNewDF) data,
    #    2. Preprocess the just-collected raw SYSMETRIC (sysPP) and ASH (ashPP2) and LABEL (labelPP) data
    #    3. Append the preprossed data to the existing SYSMETRIC (sysDF) and ASH (ashDF) 
    #       and LABEL (labelDF) dataframe
    
    # Note: Label rows are computed from sysmetric data and not collected from external system
    
    # Note: Because new sysmetric data is available every "60" seconds, but ash data is sampled
    #       every second, we use the sysmetric begin and end sample times to then bound what ash data
    #       we collect, then collect the ash data.
    
    if verbose_in:
        print("\nENTER Function getNewData")
        print("        Gathering SYSMETRIC and ASH data...")
    
    resultTF     = False
    sysNewRows   = 0
    ashNewRows   = 0
    labelNewRows = 0
    
    sysInRows   = sysDF.shape[0]
    ashInRows   = ashDF.shape[0]
    labelInRows = labelDF.shape[0]

    
    sysTF, sysNewDF = getSysmetricData(G_sysmetric_last_end_time, verbose_in)
    
    if verbose_in:
        print(sysNewDF.dtypes)
    
    sysmetric_last_beg_time = sysNewDF['SAMPLE_TIME_BEG'].min()
    sysmetric_last_end_time = sysNewDF['SAMPLE_TIME_END'].max()
    
    diff     = sysmetric_last_end_time - sysmetric_last_beg_time
    diff_sec = int(diff.days*24*60*60 + diff.seconds)
    
    if verbose_in:
        print("types: beg", type(sysmetric_last_beg_time), " end", type(sysmetric_last_beg_time) )
        print("times: beg", sysmetric_last_beg_time, "end", sysmetric_last_end_time, "diff(s)", diff_sec)
    
    ashTF, ashNewDF = getAshData(sysmetric_last_beg_time, sysmetric_last_end_time, verbose_in)
    
    
    if ashTF and sysTF:
        
        sysNewRows = sysNewDF.shape[0]
        ashNewRows = ashNewDF.shape[0]
        
        #
        # SYSMETRIC preprocessing
        #
        if verbose_in:
            print("\nPre-Processing SYSMETRIC data...")
        
        # Step 1 - Denormalize/Pivot data
        
        if verbose_in:
            print("Denormalizing sysNewDF before:", sysNewDF.shape)
            
        # The "values" will become the new "columns" value
        sysPP = sysNewDF.pivot_table(index='SAMPLE_TIME_BEG', values='METRIC_VALUE', columns=['METRIC_NAME'])
        sysPP.reset_index(inplace=True)
        
        if verbose_in:
            print("              sysPP    after:", sysPP.shape)
        
        #  Step 4 - Add the collection sequence number
        #           Append our new preprocessed sample set with the existing Sysmetric dataframe
        #           Ensure all NaN values are set to zero.
        sysPP['SEQ_NO'] = G_current_seq_num
        sysDF           = sysDF.append(sysPP)
        sysDF           = sysDF.fillna(0)
        #sysDF.reset_index(inplace=True)


        
        if verbose_in:
            print("sysDF.columns", sysDF.columns)
        
        
        #
        # ASH preprocessing
        #
        if verbose_in:
            print("\nPre-Processing ASH data...")
        
        # Step 1 - OHE
        #          We are prefixing for easier understanding
        #          dummy_na=True to ensure we have a kind of "catch all" column.
        
        ashPP = pd.DataFrame(ashNewDF)
        ashPP = pd.concat([ashPP, pd.get_dummies(ashPP['SESSION_STATE'], prefix='ash_state', dummy_na=True)], axis=1)
        ashPP = pd.concat([ashPP, pd.get_dummies(ashPP['SESSION_TYPE'], prefix='ash_type', dummy_na=True)], axis=1)
        ashPP = pd.concat([ashPP, pd.get_dummies(ashPP['WAIT_CLASS'], prefix='ash_wc', dummy_na=True)], axis=1)
        ashPP = pd.concat([ashPP, pd.get_dummies(ashPP['MODULE'], prefix='ash_mod', dummy_na=True)], axis=1)
        ashPP = pd.concat([ashPP, pd.get_dummies(ashPP['SQL_OPCODE'], prefix='ash_ocode', dummy_na=True)], axis=1)
        ashPP = ashPP.drop(columns=['SESSION_STATE','SESSION_TYPE','WAIT_CLASS','MODULE','SQL_OPCODE'])
        
        #print("ashPP.shape", ashPP.shape)
        #print(ashPP.head())
        
        
        # Step 2 - For each ASH column in this sample set, sum all the rows, them divide by the sample time (sec)
        #          This will create an average per second standardized value... like AAS
        
        #          Create the list of all the ASH columns
        #          Remove any columns we don't want to keep around
        cols = list(ashPP.columns)
        cols.remove('SAMPLE_ID')
                
        #          Do the summing and the average per second math
        #          The result will be a list (myList) of each column's average per second
        myList = []
        
        for colName in cols:
            theSum       = ashPP[colName].sum()
            theAvgPerSec = theSum / diff_sec
            
            #print(colName, theSum, round(theAvgPerSec,4))
            
            myList.append(theAvgPerSec)
        
        
        # Step 3 - Create a row dataframe. 
        #          Columns are the ASH columns, including the OHE data
        #          The row data is the average per second values
        
        # This helped me code adding a list to a dataframe as a single row
        #People_List = [['Jon','Smith','Mark','Brown']]
        #print(People_List)
        #df = pd.DataFrame (People_List,columns=['name1','name2','name3','name4'])
        #print(df)
        
        ashPP2           = pd.DataFrame([myList], columns=[cols])
        
        #  Step 4 - Add the collection sequence number
        #           Append our new preprocessed sample set with the existing ASH dataframe
        #           Ensure all NaN values are set to zero.
        
        ashPP2['SEQ_NO'] = G_current_seq_num
        ashDF            = ashDF.append(ashPP2)
        ashDF            = ashDF.fillna(0)
        #ashDF.reset_index(inplace=True)
        
        if verbose_in:
            print("ashDF.columns",ashDF.columns)
            
            
        #
        # LABEL preprocessing
        #
        # Simply uing sysmetric pre-processsed data (sysPP)
        
        if verbose_in:
            print("\nPre-Processing LABEL data...")
        
        labelPP = sysPP.copy()
        labelNewRows = labelPP.shape[0]
        labelPP['label_value'] = pow(20 * labelPP['Average Active Sessions'] + labelPP['User Calls Per Sec'], 1.34522)/100
        columns = ['SEQ_NO', 'label_value']
        labelDF = labelDF.append(labelPP[columns])
        
        if verbose_in:
            print("labelDF.columns", labelDF.columns)
            print(labelDF)
        
        
        resultTF = True
    else:
        print("Function getNewData problem. ashTF", ashTF, " sysTF", sysTF)
    
    sysMemory   = sysDF.memory_usage(deep=True).sum()
    ashMemory   = ashDF.memory_usage(deep=True).sum()
    labelMemory = labelDF.memory_usage(deep=True).sum()
    
    print("In/New/Out/MB." + "  SYS:" + str(sysInRows) +"/"+ str(sysNewRows) +"/"+ str(sysDF.shape[0]) +"/"+ str(int(sysMemory/1024/1024)) +           "  ASH:" + str(ashInRows) +"/"+ str(ashNewRows) +"/"+ str(ashDF.shape[0]) +"/"+ str(int(ashMemory/1024/1024)) +           "  LAB:" + str(labelInRows) +"/"+ str(labelNewRows) +"/"+ str(labelDF.shape[0]) +"/"+ str(int(labelMemory/1024/1024)) )
    
    if verbose_in:
        print("\nEXIT Function getNewData")
    
    return resultTF, sysDF, ashDF, labelDF


if TESTING:
    sDF = pd.DataFrame()
    aDF = pd.DataFrame()
    lDF = pd.DataFrame()
    qTF, sDF, aDF, lDF = getNewData(sDF, aDF, lDF, True)

print("\nDone.")
    


# In[ ]:


# MAIN PROGRAM (GLOBAL)

G_par_core = readParamFile(False)

data_load_on_startup_success = False

if str2bool(G_par_core["data_load_on_startup"]):
    try:
        print("Loading startup data", end="...")

        sysFN    = G_par_core["sysmetric_filename"]
        ashFN    = G_par_core["ash_filename"]
        labelFN  = G_par_core["label_filename"]
        latestFN = G_par_core["latest_collection_filename"] # not compressed

        sysDF    = pd.read_csv(sysFN, compression='gzip')
        ashDF    = pd.read_csv(ashFN, compression='gzip')
        labelDF  = pd.read_csv(labelFN, compression='gzip')
        latestDF = pd.read_csv(latestFN)

        G_current_seq_num         = latestDF['seq_num'][0]

        atime_str = latestDF['beg_time'][0]
        G_sysmetric_last_beg_time = datetime.strptime(atime_str, '%Y-%m-%d %H:%M:%S')
        
        atime = latestDF['end_time'][0]
        G_sysmetric_last_end_time = datetime.strptime(atime_str, '%Y-%m-%d %H:%M:%S')
        
        data_load_on_startup_success = True
        
        print("done.")
    
    except BaseException as e:
        print("failed:", e)

if not data_load_on_startup_success:
    
    print("NOT loading startup data. Resetting.\n")
    sysDF   = pd.DataFrame()
    ashDF   = pd.DataFrame()
    labelDF = pd.DataFrame()
    
    G_current_seq_num = 0
    G_sysmetric_last_beg_time = datetime.now() - timedelta(days=10)
    G_sysmetric_last_end_time = datetime.now() - timedelta(days=10)

print("G_current_seq_num        ", G_current_seq_num)
print("G_sysmetric_last_beg_time", G_sysmetric_last_beg_time)
print("G_sysmetric_last_end_time", G_sysmetric_last_end_time)
print()

while True:
    G_par_core = readParamFile(False) # resetting G_par_core... only place reset
    
    mainVerbose = str2bool(G_par_core["debug_detail_enable"])
        
    newDataAvailable, newBegDate, newEndDate = IsNewDataAvailable(mainVerbose)
    
    if newDataAvailable:
        G_current_seq_num = G_current_seq_num + 1
        print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), "New data begin/end time", newBegDate.strftime("%d-%b-%Y %H:%M:%S"), "/", newEndDate.strftime("%d-%b-%Y %H:%M:%S"), G_current_seq_num)
        allOK, sysDF, ashDF, labelDF = getNewData(sysDF, ashDF, labelDF, mainVerbose) # mainVerbose)
        G_sysmetric_last_beg_time = newBegDate
        G_sysmetric_last_end_time = newEndDate
        
        print("Committing", end="...")
        tic = time.perf_counter()
        
        latestData = [[G_current_seq_num, G_sysmetric_last_beg_time, G_sysmetric_last_end_time]]
        #print("latestData", latestData)
        latestDF   = pd.DataFrame(latestData, columns=['seq_num', 'beg_time', 'end_time'])
        
        sysFN    = G_par_core["sysmetric_filename"]
        ashFN    = G_par_core["ash_filename"]
        labelFN  = G_par_core["label_filename"]
        latestFN = G_par_core["latest_collection_filename"] # not compressed
        
        sysDF.to_csv(   sysFN,   header=True, index=False, compression='gzip')
        ashDF.to_csv(   ashFN,   header=True, index=False, compression='gzip')
        labelDF.to_csv( labelFN, header=True, index=False, compression='gzip')

        latestDF.to_csv(latestFN, header=True, index=False)
        
        toc = time.perf_counter()
        print(f"completed in {toc - tic:0.2f}s.")
        
    else:
         print(datetime.now().strftime("%d-%b-%Y %H:%M:%S"), "No new data.")
    
    if mainVerbose:
        print("\nSleeping...\n")
    
    time.sleep(int(G_par_core["sample_frequency_sec"]))


print("\nDone.")

