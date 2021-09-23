from django.db import models


class Human_model_img(models.Model):
    model_name = models.TextField()
    img = models.ImageField(upload_to='media/data_annotation_hm')
    mask = models.ImageField(upload_to='media/data_img_hm')

class Cloth_img(models.Model):
    cloth_name = models.TextField()
    cloth_color = models.TextField()
    cloth_img = models.ImageField(upload_to='media/cloth_img')

