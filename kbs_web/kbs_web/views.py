from django.shortcuts import render_to_response
from nn import KBSPurchaseRegressor 

def index(request):
    template = "index.html"
    kbs_regressor = KBSPurchaseRegressor()

    return render_to_response(template, {
        "regressor": kbs_regressor
    })
