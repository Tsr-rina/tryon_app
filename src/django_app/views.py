from django.shortcuts import render
from django.http import HttpResponse
from .models import Human_model_img
from .models import Cloth_img
# from .gan import GeneImg

def index(request):
    images = Human_model_img.objects.all()
    context = {'images': images}
    return render(request, 'index.html', context)

def select_cloth(request):
    if request.method == 'POST':

        

        # if 'None' in request.POST:
        #     context = {
        #         'alert':'モデルを選択してください'
        #     }
        #     return render(request, 'index',context)

        if 'm_S' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            # model_data1 = Human_model_img.objects.only()
            # model_data2 = Human_model_img.objects.values('model_name').get(id=1)
            # model_data3 = Human_model_img.objects.get(id=1)
            # print('values()', model_data)
            # print('only()', model_data1)
            # print('get()', model_data2)
            # print('values()で指定してないver',model_data3.model_name)
            # print('request.POST：', list(request.POST.values())[1])

            img = model_data[0]['img']
            mask = model_data[0]['mask']
            context = {
                'img': img,
                'name':'m_S',
                'cloth_img':cloth_img,
            }
            return render(request, 'cloth_select.html', context)

        elif 'm_M' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            img = model_data[1]['img']
            context = {
                'img': img,
                'name':'m_M',
                'cloth_img':cloth_img
            }
            return render(request, 'cloth_select.html', context)
        
        elif 'm_L' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            img = model_data[2]['img']
            context = {
                'img': img,
                'name':'m_L',
                'cloth_img':cloth_img,
            }
            return render(request, 'cloth_select.html', context)

        elif 'fm_S' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            img = model_data[3]['img']
            context = {
                'img': img,
                'name':'fm_S',
                'cloth_img':cloth_img,
            }
            return render(request, 'cloth_select.html', context)

        elif 'fm_M' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            img = model_data[4]['img']
            context = {
                'img': img,
                'name':'fm_M',
                'cloth_img':cloth_img,
            }
            return render(request, 'cloth_select.html', context)
        
        elif 'fm_L' in request.POST:
            model_data = Human_model_img.objects.values()
            cloth_img = Cloth_img.objects.values()
            img = model_data[5]['img']
            context = {
                'img': img,
                'name':'fm_L',
                'cloth_img':cloth_img,
            }
            return render(request, 'cloth_select.html', context)

def try_on(request):
    # 届くのは選択された「人体モデル」と「選択された洋服」
    if request.method == 'POST':

        if 'media/cloth_img/cloth1.jpg' in request.POST:
            # GeneImgに値を渡す
            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        if 'media/cloth_img/cloth2.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        if 'media/cloth_img/cloth3.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        if 'media/cloth_img/cloth4.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        if 'media/cloth_img/cloth5.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)
 


    


