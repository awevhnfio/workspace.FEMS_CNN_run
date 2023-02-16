'''
influxdb의 특정 태그에 값 write하기

같은 망, 다른 ip의 컴퓨터의 influxdb에 write하기 위해서는 '방화벽 8086 포트 열기'가 필요
방법은 링크 참조.
https://wiki.mcneel.com/ko/zoo/window7firewall
'''

#!/usr/bin/env python
# coding=utf-8
# from data import dt_, data, label_
##########################################################################################
from influxdb import InfluxDBClient
from copy import deepcopy
# import pprint
##########################################################################################
# import pandas as pd
import datetime


##########################################################################################
def get_ifdb(db, host='localhost', port=8086, user='', passwd=''):
    client = InfluxDBClient(host, port, user, passwd, db)
    try:
        client.create_database(db)  # db명의 db가 없으면 생성
    except:
        pass
    return client


##########################################################################################
def write_tag(ifdb, dt: datetime.datetime, tag_name, v):
    json_body = []
    # dt = datetime.datetime.now() - datetime.timedelta(hours=9)
    TIME = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    dt = dt - datetime.timedelta(hours=9)  # microsecond는 0으로

    # measurement : 태그명, fieldname : 속성명 (PV, SP 등), fieldname: 값, time: 9시간 뺀 시간
    point = {
        "measurement": tag_name,
        "tags": {
            "host": "",
        },
        "fields": {
            "PV": 0.0,
            "TIME": None,
        },
        "time": None,
    }

    np = deepcopy(point)
    np['fields']["PV"] = float(v)  # 속성명
    np['fields']["TIME"] = TIME  # 속성명
    np['time'] = dt  # time
    json_body.append(np)
    ifdb.write_points(json_body)


##########################################################################################
def my_test(ifdb, v):
    json_body = []
    dt = datetime.datetime.now() - datetime.timedelta(hours=9)
    dt = dt.replace(microsecond=0)  # microsecond는 0으로

    fieldname = 'PV'  # 속성명
    tablename = 'CNN_RESULT_FD1311'  # 태그명

    # measurement : 태그명, fieldname : 속성명 (PV, SP 등), fieldname: 값, time: 9시간 뺀 시간
    point = {
        "measurement": tablename,
        "tags": {
            "host": "",
        },
        "fields": {
            fieldname: 0.0
        },
        "time": None,
    }

    # vals = data[label]
    # for v in vals:
    np = deepcopy(point)
    np['fields'][fieldname] = float(v)
    np['time'] = dt
    json_body.append(np)
    ifdb.write_points(json_body)


##########################################################################################
def do_test():
    ifdb = get_ifdb(db='IO_INFOTROL', host='192.168.101.195')
    my_test(ifdb, v=2)


##########################################################################################
if __name__ == '__main__':
    ifdb = get_ifdb(db='IO_INFOTROL', host='192.168.101.195')
    # write_tag(ifdb, dt=datetime.datetime.now(), tag_name='TEST', v=3)

# 실행파일 생성: pyinstaller -F influx_write.py
