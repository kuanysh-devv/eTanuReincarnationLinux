from django import forms
from metadata.models import *


class CriminalCaseReasonForm(forms.ModelForm):
    class Meta:
        model = CriminalCaseReason
        fields = ['order_number', 'order_date', 'order_article']


class InvestigativeOrderReasonForm(forms.ModelForm):
    class Meta:
        model = InvestigativeOrderReason
        fields = ['order_number', 'order_date', 'document_number']


class ProsecutorInstructionReasonForm(forms.ModelForm):
    class Meta:
        model = ProsecutorInstructionReason
        fields = ['order_number', 'order_date', 'document_number', 'check_list', 'other_reason']


class InternationalOrderReasonForm(forms.ModelForm):
    class Meta:
        model = InternationalOrderReason
        fields = ['order_number', 'order_date', 'organisation']


class AfmOrderReasonForm(forms.ModelForm):
    class Meta:
        model = AfmOrderReason
        fields = ['order_number', 'order_date']


class HeadOrderReasonForm(forms.ModelForm):
    class Meta:
        model = HeadOrderReason
        fields = ['order_number', 'order_date', 'head_fio']


class OperationalInspectionReasonForm(forms.ModelForm):
    class Meta:
        model = OperationalInspectionReason
        fields = ['sphere_name']


class AnalyticalWorkReasonForm(forms.ModelForm):
    class Meta:
        model = AnalyticalWorkReason
        fields = ['theme']
