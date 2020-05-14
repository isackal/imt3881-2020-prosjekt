import kantBevGlatting as kbg
import threading
import time
from multiprocessing import Process

global_img = None

def test1():
    global global_img
    print("Running kbg")
    global_img = kbg.threadTest()
    print("Thread processing finished")

def test2():
    while True:
        time.sleep(1)
        print("hello world :D")

t1 = threading.Thread(target=test1, args=())
t2 = threading.Thread(target=test2, args=())
# t1 = Process(target=test1, args=())
# t2 = Process(target=test2, args=())

lock = threading.Lock()
t1.start()
t2.start()

t1.join()
t2.terminate()
import matplotlib.pyplot as plt
plt.show()