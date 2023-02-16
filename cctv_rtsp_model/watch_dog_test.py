"""
2022-09-13

watch dog tag에 10초 주기로 값 1 write
"""
import datetime
from user_functions.influx_write import write_tag_MV, get_ifdb
import time


ifdb = get_ifdb(db="IO_INFOTROL", host="192.168.101.109")
tag_name = "TEST2"
write_value = 1

while True:
    try:
        time_now = datetime.datetime.now()
        write_tag_MV(ifdb, dt=time_now, tag_name=tag_name, v=write_value)
    except Exception as e:
        print(e)
    time.sleep(10)
