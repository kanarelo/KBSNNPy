from django.shortcuts import render_to_response
from nn import KBSPurchaseRegressor 

def index(request):
    template = "index.html"
    return render_to_response(template, {
        "regressor": 
    })
