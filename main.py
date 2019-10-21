import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from gpiozero import LED
import math
from gpiozero import PWMLED
ledd = PWMLED(2)
red = PWMLED(14)
blue = PWMLED(15)
redr = PWMLED(3)
bluer = PWMLED(4)
camera = PiCamera()
camera.rotation = 192
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


time.sleep(0.1)




def linescount(lines):
	c = 0
	
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line[0]
			cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			c += 1
		
	
	return c


def can(image):
	image = cv2.Canny(image, 72, 150)
	return image
	
	
def leddecide(frame):
	frame = cv2.GaussianBlur(frame, (5,5), 0)
	
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	lower_blue = np.array([44,90,48])
	upper_blue = np.array([211,193,255])
	mask = cv2.inRange(hsv, lower_blue,upper_blue)
	
	_,contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	
	
	for contour in contours:
		area = cv2.contourArea(contour)
		
		
		if (area >70):
			 cv2.drawContours(frame, contour, -1, (0,255,0),1)
			 return 1
		else:
			return 0

def frequency(frame):
	if(leddecide(frame)) == 1:
		print (10)


u1 = (0,0)
d1 = (330, 400)

u2 = (320, 0)
d2 = (640, 400)





ledPin = 23

motorL = 8
motorR = 10

dc = 95  # Duty Cycle



def ledDetect():
	red.value = 0
	blue.value = 0
	ledd.value = 1
	time.sleep(1)
	ledd.value = 0
	
	

def rightt():
	red.value = 0.5
	time.sleep(1)
	red.value = 0
	
def leftt():
	blue.value = 0.5
	time.sleep(1)
	blue.value = 0
	
def forward():
    red.on()
    
    blue.on()
    

def motor():
	GPIO.output(ledPin, GPIO.HIGH)
	time.sleep(3)
	GPIO.output(ledPin, GPIO.LOW)
def Spliter():
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
		frame = frame.array
		cv2.rectangle(frame, u1, d1, (0,255,0), 5)
		cv2.rectangle(frame, u2,d2, (0,0, 255), 5)
		rect_img = frame[u1[1]: d1[1], u1[0]:d1[0]]
		rect_img2 = frame[u2[1]: d2[1], u2[0]:d2[0]]
		
		
		SR = rect_img
		SR = can(SR)
		SR = cv2.cvtColor(SR, cv2.COLOR_GRAY2RGB)
		
		frame[u1[1]: d1[1], u1[0]:d1[0] ] = SR
		
		SR1 = rect_img2
		i,SR1 = leddecide(SR1)
		print (i)
		
		frame[u2[1]: d2[1], u2[0]:d2[0]] = SR1
		
		cv2.imshow('frame', frame)
		
setState=0
ts=0
	
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	frame = frame.array
	cv2.circle(frame, (320,400), 4, (0,0,255), 3)
	#cv2.GaussianBlur(frame, (5,5), 5)
	
	#cv2.line(frame, (320, 0), (320, 480), (255,91,5), 2)
	#cv2.line(frame, (0,240), (640, 240), (0,225,0), 2)
	cv2.circle(frame, (320,400), 4, (0,0,255), 3)
	if(leddecide(frame) == 1):
		ledDetect()
		
	
	ima = cv2.GaussianBlur(frame, (5,5), 0)
	hsv = cv2.cvtColor(ima, cv2.COLOR_BGR2HSV)
	lw = np.array([0,0,100])
	uw = np.array([108,39,255])
	mask = cv2.inRange(hsv, lw,uw)
	#edge = cv2.Canny(ima,130,150)
	#rev, ima = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
	
	edge = cv2.Canny(mask,150,300)
	lines = cv2.HoughLinesP(edge, 1, np.pi/180, 100, maxLineGap = 100, minLineLength = 30)
	
	if (linescount(lines) >2):
		lines = lines.reshape(-1,4)
		
	
		print ('Three Lines')
		print (lines)
		right = lines[2]
		left = lines[0]
			
		M= tuple([int((lines[2,0] + lines[0,0])/ 2),int((lines[2,1] + lines[0,1])/ 2)])
		if (M[0] == 320):
			red.value = 1
			blue.value = 1
				
		else:
			slope = (400 - M[1]) / (320 - M[0])		
			angle = math.degrees(math.atan(slope))
			phi= 180 - angle
			print(phi)
			if (phi <90):
				red.value= abs(math.sin(phi))
				blue.value =1- abs(math.sin(phi))
				setState=1
				#time.sleep(0.5)
			if(phi>90):
				red.value = 1-abs(math.sin(phi))
				blue.value = abs(math.sin(phi))
				setState=2
				#time.sleep(0.5)
	elif (linescount(lines) == 1):
	#	lines = cv2.HoughLinesP(edge, 1, np.pi/180, 110, maxLineGap = 150, minLineLength= 5)
		lines = lines.reshape(1,4)
		print(lines)
		M = tuple([lines[0,0], lines[0,1]])
		slope = (400 - M[1]) / (320 - M[0])	
		angle = math.degrees(math.atan(slope))
		phi= 180 - angle
		if (phi >90):
			red.value= abs(math.sin(phi))
			blue.value = 1-abs(math.sin(phi))
			setState=1
		if(phi<90):
			blue.value =1- abs(math.sin(phi))
			red.value = abs(math.sin(phi))
			setState=2
	elif (linescount(lines) == 2):
	#	lines = cv2.HoughLinesP(edge, 1, np.pi/180, 110, maxLineGap = 50, minLineLength= 5)
		lines = lines.reshape(2,4)
		lines = lines[lines[:,0].argsort()]
		
		if (lines[0,0] and lines[1,0] < 320):
			slope = (400 - lines[0,1]) / (320 - lines[0,0])		
			angle = math.degrees(math.atan(slope))
			phi= 180 - angle
			print(phi)
			if (phi >90):
				red.value= abs(math.sin(phi))
				blue.value =1-abs(math.sin(phi))
				setState=2
			'''if(phi<90):
				red.value = abs(math.cos(phi))
				blue.value = abs(math.sin(phi))
				time.sleep(0.5)'''
			
		else:
			slope = (400 - lines[1,1]) / (320 - lines[1,0])		
			angle = math.degrees(math.atan(slope))
			phi= 180 - angle
			print(phi)
			if (phi <90):
				blue.value= abs(math.sin(phi))
				red.value = 1-abs(math.sin(phi))
				setState=1
				print('left')
			if(phi>90):
				blue.value =1- abs(math.sin(phi))
				red.value = abs(math.sin(phi))
				setState=2
				print('right')
				
	
		
		
	else:
		if(ts== 3):
			ts=0
			if(setState==2):
				blue.value=0.4
				red.value=0
				setState=2
				time.sleep(0.2)
				print('1 case')
			else:
				red.value=0.4
				blue.value=0
				setState=1
				time.sleep(0.2)
				print('2 case')
		else:
			ts+=1
			
			
			print('3case')
		print('zero case')	
		'''
		blue.value = 0.45
		time.sleep(0.7)
		blue.value = 0
		red.value = 0.45
		time.sleep(0.7)
		red.value = 0
		'''	
	 	
			
	cv2.imshow("frame",frame)
	
	
	
	rawCapture.truncate(0)
	key=cv2.waitKey(20)
	if key == 27:
		break
		
		
		
		

cv2.destroyAllWindows()




