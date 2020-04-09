from django.urls import path
from django.conf.urls import url
from polls import views

urlpatterns = [
    url(r'^$', views.index),
    url(r'^login_action/$',views.login_action),
    url(r'^dashboard_2/$',views.dashboard_2),
    url(r'^search_forms/$',views.search_forms),
    url(r'^search/$',views.search),
    url(r'^predict_1/$',views.predict_1),
    url(r'^predict_2/$',views.predict_2),
    path('mailbox/',views.mailbox),
    path('introduction/',views.introduction),
    path('graph_flot/',views.graph_flot),
    path('map_page/',views.map_page),
    path('test/',views.test),
    path('predict/',views.predict),
    path('predict_m/',views.predict_m),
]

 