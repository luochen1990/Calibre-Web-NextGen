"""Synthetic omitted-row recovery cannot duplicate another unit's occurrence."""
import pytest
from cps.services.reflow.vision_pipeline import _remove_foreign_word_runs

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('proof',['owned','elsewhere','ambiguous','unowned'])
def test_unbound_tail_is_removed_only_at_a_unique_own_to_foreign_boundary(proof):
    optical=[dict(text=t,bbox=[10+i*25,20,30+i*25,32])for i,t in enumerate('The garden stays green today.'.split())]
    boxes={0:optical[0]['bbox'],1:optical[1]['bbox']}
    if proof=='elsewhere':boxes[1]=[35,80,55,92]
    if proof=='ambiguous':optical=optical+optical[1:]
    regions={'other':[[60,18,160,34]]}if proof!='unowned'else{}
    result=dict(tokens='The garden stays green today.'.split(),boxes=boxes,marked={2,3,4},
        word_notes={i:dict(status='UNRESOLVED')for i in [2,3,4]},guard=[])
    _remove_foreign_word_runs(result,'first',{'blocks':[{'id':'first'},{'id':'other'}]},optical,regions)
    assert result['tokens']==(['The','garden']if proof=='owned'else'The garden stays green today.'.split())
    if proof=='owned':
        assert not result['marked']
        assert result['guard'][0]['runs'][0]['owner']=='other'
    else:assert not result['guard']
