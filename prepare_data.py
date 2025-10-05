import os

os.system("wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Orkut.tar.gz")
os.system("tar -zxvf com-Orkut.tar.gz")
os.system("cp com-Orkut/com-Orkut.mtx /dev/shm/")
for _ in range(4):
    os.system(f"mkdir /dev/shm/image_input_{_}")
    os.system(f"mkdir /dev/shm/image_output_{_}")
    os.system(f"cp side_task/image_resize_watermark_side_task/input_images/img4.jpg /dev/shm/image_input_{_}/")
