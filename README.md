# BigData_Project: Blur faces and license plates using deep-learning
Complete YOLO v3 TensorFlow implementation. Support training on your own dataset.

This project is worked with 서수진, 원동호, 이민지, 전여진

The code is built on Ubuntu 18.04 enviroment(python_3.6, tensorflow_1.14.0, django_2.0.13,
opencv-python_4.1.0.25 and ngrok_2.3.34) with GTX-1050ti.

## Getting Started

Our weight can download on [here]()

1. Run the server in your project directory

```
$ python3 manage.py runserver
```

2. Activate server using ngrok on other terminal

```
$ ./ngrok http [port number](maybe 8000)
```
   
3. Copy the address of 'Fowarding' row (ex. d61b0f6fngrok.io)
   * Modify setting.py in FirstProject directory
   * Paste the address to ALLOWED_HOSTS in 28th row
 
4. Open the address on your browser
   * If you want blur the image, click ![image_icon](./blog/static/img/image_icon.png) icon
   * After click "파일선택" button, select the image to blur
   * Click convert button
   * If you want to download blured image, click ![download_icon](./blog/static/img/download_icon.png) icon
   * If you want to go home, click ![home_icon](./blog/static/img/home_icon.png) icon
   * We don't produce to blur the **video**
