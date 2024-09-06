from django.contrib import admin

from metadata.models import *

admin.site.register(Person)
admin.site.register(Account)
admin.site.register(Gallery)
admin.site.register(SearchHistory)
admin.site.register(AnalyticalWorkReason)
admin.site.register(AfmOrderReason)
admin.site.register(ProsecutorInstructionReason)
admin.site.register(InternationalOrderReason)
admin.site.register(OperationalInspectionReason)
admin.site.register(HeadOrderReason)
