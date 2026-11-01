"""Documented Gemini calendar-day RPD accounting without deleting ledger entries.

Source: https://ai.google.dev/gemini-api/docs/rate-limits
RPD resets midnight America/Los_Angeles; TPM/RPM and TPD remain conservative.
The original provider ledger, flock, atomic writes, reconciliation and pauses remain.
"""
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import time


def pacific_window(now):
    local=datetime.fromtimestamp(now,ZoneInfo('America/Los_Angeles'))
    start=local.replace(hour=0,minute=0,second=0,microsecond=0)
    return start.timestamp(),(start+timedelta(days=1)).timestamp()


def install(providers):
    if hasattr(providers,'fast_original_quotas'):return
    original=providers.Quotas
    class DocumentedGeminiQuotas(original):
        def reserve(self,ident,tokens,now=None):
            if self.provider!='gemini':return super().reserve(ident,tokens,now)
            now=time.time() if now is None else now;start,reset=pacific_window(now)
            if tokens>min(self.limits['tpm'],self.limits['tpd']):
                raise providers.SafeAPIError('Request exceeds verified free token capacity; no truncation permitted')
            def action(state):
                if state['blocked_until']>now:raise providers.QuotaPause(self.provider,state['blocked_until'])
                minute=[e for e in state['entries'] if e['time']>now-60]
                rolling=[e for e in state['entries'] if e['time']>now-86400]
                day=[e for e in state['entries'] if e['time']>=start]
                if len(minute)>=self.limits['rpm'] or sum(e['quota_tokens'] for e in minute)+tokens>self.limits['tpm']:
                    raise providers.QuotaPause(self.provider,min(e['time'] for e in minute)+61)
                if len(day)>=self.limits['rpd']:raise providers.QuotaPause(self.provider,reset+1)
                if sum(providers.daily_tokens(e) for e in rolling)+tokens>self.limits['tpd']:
                    raise providers.QuotaPause(self.provider,min(e['time'] for e in rolling)+86401)
                index=len(state['entries'])
                state['entries'].append({'request_sha256':ident,'time':now,'quota_tokens':tokens,'status':'reserved'})
                return index
            return self.update(action)
    providers.fast_original_quotas=original;providers.Quotas=DocumentedGeminiQuotas
