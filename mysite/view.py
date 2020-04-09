from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect   
def index(request):
    return render(request, 'index.html')                            
def login_action(request):
    if request.method == 'POST':                                            #判断是否为post提交方式
        username = request.POST.get('username', '')                         #通过post.get()方法获取输入的用户名及密码
        password =request.POST.get('password', '')

        if username == 'zyx' and password == '123':                        #判断用户名及密码是否正确
            return HttpResponseRedirect('/dashboard_2/')                    #如果正确，（这里调用另一个函数，实现登陆成功页面独立，使用HttpResponseRedirect()方法实现
        else:
            return render(request,'index.html',{'error':'用户名或密码错误！'})#不正确，通过render(request,"index.html")方法在error标签处显示错误提示
def dashboard_2(request):                                                            #该函数定义的是成功页面的提示页面
    return render(request,"dashboard_2.html")                                #在上面的函数判断用户名密码正确后在显示该页面，指定到event_manage.html,切换到一个新的html页面

def graph_flot(request):
    return render(request,"graph_flot.html")

def empty_page(request):
    return render(request,"empty_page.html")