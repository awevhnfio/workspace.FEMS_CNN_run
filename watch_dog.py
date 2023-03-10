"""
2022-09-13

watch dog tag에 10초 주기로 값 1 write
"""
import datetime
from user_functions.influx_write import write_tag_MV, get_ifdb
import time

# influxdb 불러오기
ifdb = get_ifdb(db="IO_INFOTROL", host="win-hn1itrpjqk8")

# write할 tag명, 값 설정
tag_name = "ARE1336_AI_WTD_RE"
write_value = 1

# 10초 간격으로 무한 반복
while True:
    try:
        time_now = datetime.datetime.now()
        write_tag_MV(ifdb, dt=time_now, tag_name=tag_name, v=write_value)
    except Exception as e:
        print(e)
    time.sleep(10)
