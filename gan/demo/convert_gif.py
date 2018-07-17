import matplotlib.pyplot as plt
import imageio,os
images = []
filenames=[]
for i in range(100):
    filenames.append("demo_Epoch"+str(i)+".jpg")
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('source.gif', images,duration=0.2)
