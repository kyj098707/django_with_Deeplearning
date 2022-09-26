from django.shortcuts import render
from .clf import load_clf_model, do_predict
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import PIL
import os, datetime, random
from latexproject.settings import MEDIA_ROOT 


def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name

def home(request):
    if request.POST:
        model = load_clf_model()
        img_classes = ['Dog','Cat']
        file1_path, file1_name = handle_uploaded_file(request.FILES['file1'])
        value, image_one_class = do_predict(model,file1_path)
        img_class = img_classes[image_one_class]

        return render(request,"classifier/base.html",{"Class":img_class,"post":True,"img1src":file1_name})
    return render(request, "classifier/base.html")
