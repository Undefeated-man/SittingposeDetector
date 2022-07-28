import time

time.sleep(60)
with open("mask", "w") as f:
    f.write("F")
print("done")