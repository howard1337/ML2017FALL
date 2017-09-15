from PIL import Image
import sys

im_1 = Image.open(sys.argv[1])
width_1 ,height_1 = im_1.size


for x in range(0,width_1):
	for y in range(0,height_1):
		tmp = im_1.load()	
		im_1.putpixel((x,y),tuple(t//2 for t in tmp[x,y]))
im_1.save("Q2.jpg")	