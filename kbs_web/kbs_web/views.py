from django.shortcuts import redirect, render_to_response
from nn import KBSPurchaseRegressor 

def index(request):
    template = "index.html"
    kbs_regressor = KBSPurchaseRegressor()

    if request.method == "POST":
        context = {}
    else:
        context = {}

    return render_to_response(template, context)

def retrain(request):
    if request.method == "POST":
        kbs_regressor = KBSPurchaseRegressor(retrain=True)
        kbs_regressor.predict()

    return redirect(index)
