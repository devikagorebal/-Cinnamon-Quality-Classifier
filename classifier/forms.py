from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


def _num(label, step="0.01"):
    return forms.FloatField(label=label, min_value=0,
                            widget=forms.NumberInput(attrs={"step": step, "placeholder": "0.00"}))


class PredictForm(forms.Form):
    moisture = _num("Moisture (%)")
    ash = _num("Total Ash (%)")
    volatile_oil = _num("Volatile Oil (ml/100g)")
    acid_insoluble_ash = _num("Acid Insoluble Ash (%)")
    chromium = _num("Chromium (mg/kg)")
    coumarin = _num("Coumarin (mg/kg)")
