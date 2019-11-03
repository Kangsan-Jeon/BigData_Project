# BigData_Project: Blur faces and license plates using deep-learning

This project is worked with 서수진, 원동호, 이민지, 전여진

The code is built on Ubuntu 18.04 enviroment(python_3.6, tensorflow_1.14.0, django_2.0.13,
opencv-python_4.1.0.25 and ngrok_2.3.34) with GTX-1050ti.

We trained face and license plate images with [YOLO-v3](https://pjreddie.com/darknet/yolo/)
> face: [wider face](http://shuoyang1213.me/WIDERFACE/)(3226), license plate: [AOLP](http://aolpr.ntust.edu.tw/lab/download.html)(2049), [MediaLAb LPR](http://www.medialab.ntua.gr/research/LPRdatabase.html)(590) and self collection(327))

## Getting Started

Our weight can download on [here](https://drive.google.com/open?id=1WWZZ-rciDmJCv4CBwOaFD4l-ZMVb85ZF)

1. Run the server in your project directory

```
$ python3 manage.py runserver
```

2. Activate server using ngrok on other terminal
   * Copy the address of 'Fowarding' row on terminal after execute below code (ex. d61b0f6fngrok.io)

```
$ ./ngrok http [port number] (maybe 8000)
```

3. Modify setting.py in FirstProject directory
   * Paste the address to ALLOWED_HOSTS in 28th row
 
4. Blur a image
   * Open the address on your browser
   * If you want blur the image, click ![image_icon](./blog/static/img/image_icon.png) icon
   * After click "파일선택" button, select the image to blur
   * Click convert button
   * If you want to download blured image, click ![download_icon](./blog/static/img/download_icon.png) icon
   * If you want to go home, click ![home_icon](./blog/static/img/home_icon.png) icon
   * We don't produce to blur the **video**
