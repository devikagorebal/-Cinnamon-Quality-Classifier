from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from .forms import RegisterForm, PredictForm

ARTIFACTS = Path(settings.BASE_DIR) / "ml" / "artifacts.joblib"


@lru_cache(maxsize=1)
def load_artifacts():
    return joblib.load(ARTIFACTS)


def register(request):
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.save())
        messages.success(request, "Account created. Welcome!")
        return redirect("predict")
    return render(request, "classifier/register.html", {"form": form})


@login_required
def predict(request):
    form = PredictForm(request.POST or None)
    ctx = {"form": form}
    if request.method == "POST" and form.is_valid():
        if not ARTIFACTS.exists():
            messages.error(request, "Model not trained yet. Run: python train_model.py")
            return render(request, "classifier/predict.html", ctx)
        art = load_artifacts()
        row = pd.DataFrame([form.cleaned_data])[art["features"]]
        scaled = art["scaler"].transform(row)
        results = []
        for name, model in art["models"].items():
            label = art["encoder"].inverse_transform(model.predict(scaled))[0]
            results.append({"name": name, "prediction": label,
                            "accuracy": art["accuracies"][name], "is_best": name == art["best"]})
        results.sort(key=lambda r: -r["accuracy"])
        best = next(r for r in results if r["is_best"])
        ctx.update(results=results, final=best["prediction"], best=best["name"])
    return render(request, "classifier/predict.html", ctx)
