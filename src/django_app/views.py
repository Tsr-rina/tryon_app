from django.shortcuts import render
from django.http import HttpResponse
from .models import Human_model_img
from .models import Cloth_img
from django_app.gan import CallNetWork
import cv2
import numpy as np
from PIL import Image
import io
import base64
from django_app.senga import makecounter
from django_app.overlay import overlayImage

def index(request):
    return render(request, 'index.html')

def select_model(request):
    if request.method == 'POST':
        images = Human_model_img.objects.all()
        context = {'images': images}
        return render(request, 'model_select.html', context)   

def select_cloth(request):
    if request.method == 'POST':
        if (not request.POST.get("m_S")) and (not request.POST.get("m_M")) and (not request.POST.get("m_L")) and (not request.POST.get("fm_S")) and (not request.POST.get("fm_M")) and (not request.POST.get("fm_L")):
            images = Human_model_img.objects.all()
            context = {
                'message':'モデルを1つ選択してください',
                'images': images
            }  
            return render(request, "model_select.html", context)

        # if 'None' in request.POST:
        #     context = {
        #         'alert':'モデルを選択してください'
        #     }
        #     return render(request, 'index',context)

        elif 'm_S' in request.POST:
            keys_list = request.POST.keys()
            print(keys_list)
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
    # ganモジュールに送信→全身画像,マスク画像,洋服
    if request.method == 'POST': 
        keys_list = request.POST.keys()
        keys_list = list(keys_list)
        # if (not request.POST.get("media/cloth_img/cloth1.jpg")) and (not request.POST.get("media/cloth_img/cloth2.jpg")) and (not request.POST.get("media/cloth_img/cloth3.jpg")) and (not request.POST.get("media/cloth_img/cloth4.jpg")) and (not request.POST.get("media/cloth_img/cloth5.jpg")) and (not request.POST.get("media/cloth_img/cloth6.jpg")):
        if (keys_list[2] != 'media/cloth_img/cloth1.jpg') and (keys_list[2] != 'media/cloth_img/cloth2.jpg') and (keys_list[2] != 'media/cloth_img/cloth3.jpg') and (keys_list[2] != 'media/cloth_img/cloth4.jpg') and (keys_list[2] != 'media/cloth_img/cloth5.jpg') and (keys_list[2] != 'media/cloth_img/cloth6.jpg'):
            
            keys_list = request.POST.keys()
            keys_list = list(keys_list)
            print(keys_list)
            # # 選択した服
            # print(keys_list[2])

            # 人体モデル
            hm_img = Human_model_img.objects.values('img').filter(model_name=keys_list[0])
            img_path = list(list(hm_img)[0].values())
            # print(hm_img)
            # print(img_path[0])

            cloth_img = Cloth_img.objects.all()
            context = {
                'message':'洋服を1つ選択してください',
                'cloth_img': cloth_img,
                'img':img_path[0],
                'name':keys_list[0]
            }
            return render(request, "cloth_select.html", context)

        elif 'media/cloth_img/cloth1.jpg' in request.POST:
            # keys_list = request.POST.keys()
            # keys_list = list(keys_list)
            print("通過4")

            # 選択した洋服
            cloth_path = keys_list[2]
            # print("cloth_path:{}".format(cloth_path))
            cloth_path = "./media/"+str(cloth_path)

            # 選択した人体モデルのマスク画像取得
            hm_img_mask = Human_model_img.objects.values('mask').filter(model_name=keys_list[0])
            mask_path = list(list(hm_img_mask)[0].values())
            mask_path = "./media/"+str(mask_path[0])
            # print("mask_path:{}".format(mask_path))

            # 人体モデル
            hm_img = Human_model_img.objects.values('img').filter(model_name=keys_list[0])
            hm_path = list(list(hm_img)[0].values())
            hm_path = "./media/"+str(hm_path[0])

            # ここに画像処理
            mask_img = cv2.imread(mask_path)
            print("mask_img.shape:{}".format(mask_img.shape))
            resized_mask = cv2.resize(mask_img, dsize=(192,256))
            print("resized_mask:{}".format(resized_mask.shape))

            # 画像抽出paste
            get_mask = resized_mask[45:128, 57:130]
            get_mask_shape = get_mask.shape
            print("get_mask:{}".format(get_mask.shape))
            get_mask = cv2.resize(get_mask, dsize=(192,256))
            print("抽出後のリサイズ:{}".format(get_mask.shape))

            # 洋服の線画past
            cloth = makecounter(cloth_path)
            # cloth = cv2.imread()
            print("cloth.shape:{}".format(cloth.shape))
            resized_cloth = cv2.resize(cloth, dsize=(192,256))
            print("resized_cloth:{}".format(resized_cloth.shape))

            # オーバーレイ
            black = [0, 0, 0]
            hotpink = [255, 105, 180]
            get_mask[np.where((get_mask == black).all(axis=2))] = hotpink
            get_mask = cv2.cvtColor(get_mask, cv2.COLOR_RGB2BGRA)
            get_mask[:, :,3] = np.where(np.all(get_mask == (255, 255,255,255), axis=-1), 0, 255)  # 白色のみTrueを返し、Alphaを0にする
            colored_image = overlayImage(cloth, get_mask, (0, 0))   

            # GANへ
            # 紫色をしろ色にする処理を書く

            # GANの結果を受け取り
            # なのでcolored_imageが変更の可能性あり
            made = cv2.resize(colored_image, dsize=(get_mask_shape[1], get_mask_shape[0]))
            made = cv2.cvtColor(made, cv2.COLOR_BGRA2BGR)
            print("made.shape:{}".format(made.shape))
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, dsize=(192, 256))
            print("mask.shape:{}".format(mask.shape))
            made_plot = [[0,0],[0,get_mask_shape[0]],[get_mask_shape[1],get_mask_shape[0]], [get_mask_shape[1],0]]
            mask_plot = [[57,45],[57,128],[130, 128], [130, 45]]

            # バウンディングボックスで切り出し
            src_pts_arr = np.array(made_plot, dtype=np.float32)
            dst_pts_arr = np.array(mask_plot, dtype=np.float32)

            src_rect = cv2.boundingRect(src_pts_arr)
            dst_rect = cv2.boundingRect(dst_pts_arr)

            print("src_rect:{}".format(src_rect))
            print("dst_rect:{}".format(dst_rect))


            made_crop = made[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
            mask_crop = mask[dst_rect[1]:dst_rect[1] + dst_rect[3]-1, dst_rect[0]:dst_rect[0] + dst_rect[2]-1]
            print(made_crop.shape)
            print(mask_crop.shape)
            made_pts_crop = src_pts_arr - src_rect[:2]
            mask_pts_crop = dst_pts_arr - dst_rect[:2]

            print("made_crop:{}".format(made_pts_crop))
            print("mask_crop:{}".format(mask_pts_crop))

            masker = np.zeros_like(mask_crop, dtype=np.float32)
            cv2.fillConvexPoly(masker, mask_pts_crop.astype(np.int), (1.0,1.0,1.0), cv2.LINE_AA)
            dst_crop_merge = made_crop * mask_crop + mask_crop * (1 - mask_crop)

            mask[dst_rect[1]:dst_rect[1]+dst_rect[3]-1, dst_rect[0]:dst_rect[0]+dst_rect[2]-1] = dst_crop_merge

            # GAN済のマスク画像が生成できたら人体モデルと合成する
            mask = cv2.resize(mask, dsize=(192,256))
            black = [0, 0, 0]
            hotpink = [255, 105, 180]
            mask[np.where((mask == black).all(axis=2))] = hotpink
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGRA)
            mask[:, :,3] = np.where(np.all(mask == (180,105,255,255), axis=-1), 0, 255)  # 白色のみTrueを返し、Alphaを0にする
            # 人体モデル読み込み
            hm = cv2.imread(hm_path)
            hmed = cv2.resize(hm, dsize=(192,256))

            all_got = overlayImage(hmed, mask, (0, 0))

            # OpenCV→PIL変換
            image_cv = cv2.cvtColor(all_got, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_cv)
            image_pil = image_pil.convert('RGB')

            # 画像をhtmlに持っていく準備
            buffer = io.BytesIO()
            image_pil.save(buffer, format="PNG")

            base64Img = base64.b64encode(buffer.getvalue()).decode().replace("'","")

            # 人体モデル
            # hm_img = Human_model_img.objects.values('img').filter(model_name=keys_list[0])
            # img_path = list(list(hm_img)[0].values())[0]

            # # マスク画像
            # hm_mask = Human_model_img.objects.values('mask').filter(model_name=keys_list[0])
            # mask_path = list(list(hm_mask)[0].values())[0]

            # print("ganmodule行ってきます")
            # image_make = CallNetWork(str(img_path), str(mask_path), str(cloth_path))
            # made = image_make.forward()
            # print("戻ってきました")
            # print(made.shape)


            context = {
                'hello': 'Hello World!',
                'cloth_path':'media/cloth_img/cloth1.jpg',
                'mask_path':mask_path,
                'base64Img':base64Img,
            }
            return render(request, 'result.html', context)

        elif 'media/cloth_img/cloth2.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        elif 'media/cloth_img/cloth3.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        elif 'media/cloth_img/cloth4.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)

        elif 'media/cloth_img/cloth5.jpg' in request.POST:

            context = {
                'hello': 'Hello World!',
            }
            return render(request, 'result.html', context)
 


    


