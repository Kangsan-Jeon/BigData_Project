from django import forms
from.models import Video

class UploadImageForm(forms.Form):
    #file = forms.FileField()
    image = forms.ImageField()

class VideoForm(forms.ModelForm):
    class Meta:
        model= Video
        fields= ["name", "videofile"]