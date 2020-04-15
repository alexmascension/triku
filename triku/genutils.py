import os

def get_cpu_count():
    # adapted from stackoverflow.com/questions/1006289
    workers = os.cpu_count()

    if 'sched_getaffinity' in dir(os):
        workers = len(os.sched_getaffinity(0))

    return workers
    