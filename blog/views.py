import os
import numpy as np
import pickle
import platform
import cv2

from django.shortcuts import render, redirect
from .forms import UploadImageForm
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from YOLOv3_TensorFlow.test_single_image_mosaik import imgprocessing

from django.shortcuts import render
from .models import Video
from .forms import VideoForm

def post_list(request):
    return render(request, 'blog/post_list.html', {})


def uimage(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)        # 이미지를 업로드 하기 위한 폼
        if form.is_valid():
            myfile = request.FILES['image']
            fs = FileSystemStorage()                               # 이미지 파일을 저장하는 코드
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = os.getcwd() + fs.url(filename)                  # 이미지 업로드
            result_url = imgprocessing(uploaded_file_url)
            print("결과 저장된 파일 경로" + result_url)
            if os.path.isfile(result_url):
                result_url = result_url[1:]
                return render(request, 'blog/uimage.html', {'form': form, 'result_url': result_url})
            
#             return render(request, 'blog/uimage.html', {'form': form, 'uploaded_file_url': uploaded_file_url})   # 그 결과를 html에 반환
    else:
        form = UploadImageForm()
        return render(request, 'blog/uimage.html', {'form': form})


def uvideo(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)  # 이미지를 업로드 하기 위한 폼
        if form.is_valid():
            myfile = request.FILES['image']
            fs = FileSystemStorage()  # 이미지 파일을 저장하는 코드
            print(myfile.name)
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = os.getcwd() + fs.url(filename)  # 이미지 업로드
            result_url = imgprocessing(uploaded_file_url)
            print("결과 저장된 파일 경로" + result_url)
            if os.path.isfile(result_url):
                result_url = result_url[1:]
                return render(request, 'blog/uvideo.html', {'form': form, 'result_url': result_url})

    #             return render(request, 'blog/uimage.html', {'form': form, 'uploaded_file_url': uploaded_file_url})   # 그 결과를 html에 반환
    else:
        form = VideoForm()
        return render(request, 'blog/uvideo.html', {'form': form})


# def showvideo(request):
#     lastvideo = Video.objects.last()
#
#     videofile = lastvideo.videofile
#
#     form = VideoForm(request.POST or None, request.FILES or None)
#     if form.is_valid():
#         form.save()
#
#     context = {'videofile': videofile,
#                'form': form
#                }
#
#     return render(request, 'Blog/videos.html', context)