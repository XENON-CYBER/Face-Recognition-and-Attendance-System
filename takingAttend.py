import schedule
import time

def task():
    print("this is oding thing")

schedule.every(5).seconds.do(task)

while True:
    schedule.run_pending()
    time.sleep(1)
