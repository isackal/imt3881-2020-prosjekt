import threading
import time

def test1():
    while True:
        time.sleep(0.1)
        print("hello :)")

def test2():
    while True:
        time.sleep(0.11)
        print("world :D")

t1 = threading.Thread(target=test1, args=())
t2 = threading.Thread(target=test2, args=())

lock = threading.Lock()
t1.start()
t2.start()

while True:
    k = input("Input 1: ")
    t1._stop()
    t2._stop()
    k = input("Input 2: " )