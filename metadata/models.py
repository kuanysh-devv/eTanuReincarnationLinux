from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from jsonfield import JSONField


class Person(models.Model):
    iin = models.CharField(max_length=255, verbose_name=_("iin"), default="", null=True, blank=True)
    firstname = models.CharField(max_length=255, verbose_name=_("firstName"), default="", null=True, blank=True)
    surname = models.CharField(max_length=255, verbose_name=_("surname"), default="", null=True, blank=True)
    patronymic = models.CharField(max_length=255, verbose_name=_("patronymic"), default="", null=True, blank=True)
    birthdate = models.DateField(verbose_name=_("birthdate"), null=True, blank=True)

    def __str__(self):
        return str(self.surname) + " " + str(self.firstname) + " " + str(self.patronymic)

    class Meta:
        verbose_name = _("Person")
        verbose_name_plural = _("Person instances")


class Gallery(models.Model):
    vector_id = models.CharField(max_length=50, primary_key=True, verbose_name=_("Vector Id"))
    photo = models.TextField(verbose_name=_("Photo"))
    personId = models.ForeignKey(Person, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.personId.surname) + " " + str(self.personId.firstname) + " " + str(
            self.personId.patronymic) + str(self.photo)

    class Meta:
        verbose_name = _("Gallery")
        verbose_name_plural = _("Gallery instances")


class Account(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userAccount')
    firstname = models.CharField(max_length=255, verbose_name=_("firstName"), default="", null=True, blank=True)
    surname = models.CharField(max_length=255, verbose_name=_("surname"), default="", null=True, blank=True)
    patronymic = models.CharField(max_length=255, verbose_name=_("patronymic"), default="", null=True, blank=True)
    role_id = models.CharField(max_length=50)
    face_vector_id = models.CharField(max_length=2555, verbose_name=_("Face Vector Id"), null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Created at"), null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Updated at"), null=True, blank=True)

    def __str__(self):
        return f"{self.user.username}"

    class Meta:
        verbose_name = _("Account")
        verbose_name_plural = _("Accounts")


class SearchReason(models.TextChoices):
    CRIMINAL_CASE = 'CRIMINAL_CASE', _("Criminal case")
    INVESTIGATIVE_ORDERS = 'INVESTIGATIVE_ORDER', _("Investigative orders")
    PROSECUTOR_INSTRUCTIONS = 'PROSECUTOR_INSTRUCTION', _("Prosecutor instructions")
    INTERNATIONAL_ORDERS = 'INTERNATIONAL_ORDER', _("International orders")
    AFM_ORDERS = 'AFM_ORDER', _("Afm orders")
    HEAD_ORDERS = 'HEAD_ORDER', _("Head orders")
    OPERATIONAL_INSPECTIONS = 'OPERATIONAL_INSPECTION', _("Operational inspections")
    ANALYTICAL_WORK = 'ANALYTICAL_WORK', _("Analytical work")


class SearchHistory(models.Model):
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='account')
    searchedPhoto = models.TextField(verbose_name=_("Photo"))
    created_at = models.DateTimeField(verbose_name=_("Created at"))
    reason = models.CharField(max_length=500, choices=SearchReason.choices, verbose_name=_("Search Reason"),
                              null=True, blank=True)

    def __str__(self):
        return f"{self.account.user.username} {self.created_at}"

    class Meta:
        verbose_name = _("History")
        verbose_name_plural = _("Histories")


class CriminalCaseReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='criminal_case_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("Order number"))
    order_date = models.DateField(verbose_name=_("Order date"))
    order_article = models.CharField(max_length=255, verbose_name=_("Order article"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("CriminalCaseReason")
        verbose_name_plural = _("CriminalCaseReasons")


class InvestigativeOrderReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='investigative_order_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("Investigative order number"))
    order_date = models.DateField(verbose_name=_("Investigative order date"))
    document_number = models.CharField(max_length=255, verbose_name=_("Investigative document number"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("InvestigativeOrderReason")
        verbose_name_plural = _("InvestigativeOrderReasons")


class ProsecutorInstructionReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='prosecutor_order_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("prosecutor order number"))
    order_date = models.DateField(verbose_name=_("prosecutor order date"))
    document_number = models.CharField(max_length=255, verbose_name=_("prosecutor document number"))
    check_list = models.CharField(max_length=255, verbose_name=_("prosecutor check list"), null=True, blank=True)
    other_reason = models.CharField(max_length=255, verbose_name=_("prosecutor other reason"), null=True, blank=True)

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("ProsecutorInstructionReason")
        verbose_name_plural = _("ProsecutorInstructionReasons")


class InternationalOrderReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='international_order_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("international order number"))
    order_date = models.DateField(verbose_name=_("international order date"))
    organisation = models.CharField(max_length=255, verbose_name=_("international organisation"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("InternationalOrderReason")
        verbose_name_plural = _("InternationalOrderReasons")


class AfmOrderReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='afm_order_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("afm order number"))
    order_date = models.DateField(verbose_name=_("afm order date"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("AfmOrderReason")
        verbose_name_plural = _("AfmOrderReasons")


class HeadOrderReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='head_order_reason')
    order_number = models.CharField(max_length=255, verbose_name=_("head order number"))
    order_date = models.DateField(verbose_name=_("head order date"))
    head_fio = models.CharField(max_length=255, verbose_name=_("head fio"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("HeadOrderReason")
        verbose_name_plural = _("HeadOrderReasons")


class OperationalInspectionReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='operational_inspection_reason')
    sphere_name = models.CharField(max_length=255, verbose_name=_("sphere name"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("OperationalInspectionReason")
        verbose_name_plural = _("OperationalInspectionReasons")


class AnalyticalWorkReason(models.Model):
    search_history = models.OneToOneField(SearchHistory, on_delete=models.CASCADE,
                                          related_name='analytical_work_reason')
    theme = models.CharField(max_length=255, verbose_name=_("analytical work theme"))

    def __str__(self):
        return f"{self.search_history.id}"

    class Meta:
        verbose_name = _("AnalyticalWorkReason")
        verbose_name_plural = _("AnalyticalWorkReasons")
