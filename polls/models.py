# Create your models here.
#！/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime
from django.db import models
from django.utils import timezone

def __str__(self):
    return self.Choice_text

class user_info(models.Model):
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=20)

class address_info(models.Model):
    crossID = models.IntegerField()
    longitude = models.FloatField()
    latitude = models.FloatField()
    detectorNum = models.IntegerField()
 
class crossinfo_8(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_9(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_13(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_14(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_15(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_19(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_20(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_22(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_27(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_68(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_70(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_71(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_79(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_106(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_157(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_191(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_194(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_195(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_199(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_201(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_263(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_319(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_320(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_321(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_336(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_405(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_471(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_472(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_474(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_475(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_564(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_565(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_647(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_754(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_755(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_920(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class crossinfo_968(models.Model):
    detectorID = models.IntegerField()
    DS = models.IntegerField()
    actualVolume = models.IntegerField()
    time = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    PL = models.FloatField()

class pl_22(models.Model):
    pl = models.IntegerField()
    A = models.CharField(max_length=200)
    B = models.CharField(max_length=200)
    C = models.CharField(max_length=200)
    D = models.CharField(max_length=200)
    E = models.CharField(max_length=200)
    F = models.CharField(max_length=200)
    G = models.CharField(max_length=200)

class pl_8(models.Model):
    pl = models.IntegerField()
    A = models.CharField(max_length=200)
    B = models.CharField(max_length=200)
    C = models.CharField(max_length=200)
    D = models.CharField(max_length=200)
    E = models.CharField(max_length=200)
    F = models.CharField(max_length=200)
    G = models.CharField(max_length=200)

class pl_14(models.Model):
    pl = models.IntegerField()
    A = models.CharField(max_length=200)
    B = models.CharField(max_length=200)
    C = models.CharField(max_length=200)
    D = models.CharField(max_length=200)
    E = models.CharField(max_length=200)
    F = models.CharField(max_length=200)
    G = models.CharField(max_length=200)

class pl_19(models.Model):
    pl = models.IntegerField()
    A = models.CharField(max_length=200)
    B = models.CharField(max_length=200)
    C = models.CharField(max_length=200)
    D = models.CharField(max_length=200)
    E = models.CharField(max_length=200)
    F = models.CharField(max_length=200)
    G = models.CharField(max_length=200)