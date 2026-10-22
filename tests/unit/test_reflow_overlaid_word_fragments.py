"""Synthetic overlapping text-layer aliases are one printed occurrence."""
import pytest
from cps.services.reflow import pixel_adjudication as pa

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('separate',[False,True])
def test_complete_pixel_word_cannot_publish_an_overlaid_extra_letter(separate):
    whole=dict(text='garden',bbox=[40,20,82,32],confidence=99)
    fragment=dict(text='de',bbox=[60,20,70,32]if not separate else[90,20,100,32],confidence=99)
    source=[dict(text='The',bbox=[10,20,30,32],confidence=99),whole,fragment,
            dict(text='stays',bbox=[110,20,140,32],confidence=99)]
    witness=pa.CropWitness([dict(w)for w in source if w is not fragment])
    result=pa.adjudicate(['The','garden','stays'],source,witness)
    assert result['tokens']==(['The','garden','de','stays']if separate else ['The','garden','stays'])
    if not separate:assert not result['marked'],'The same original ink is not a new source word or uncertainty'


@pytest.mark.parametrize('fragment_text', ['arde', 'arden'])
@pytest.mark.parametrize('evidence', ['complete', 'separate', 'unconfirmed'])
def test_complete_pixel_word_owns_a_longer_overlaid_fragment(fragment_text, evidence):
    whole = dict(text='garden', bbox=[40, 20, 82, 32], confidence=99)
    fragment = dict(text=fragment_text,
                    bbox=[50, 22, 72, 30] if evidence != 'separate' else [90, 20, 112, 32],
                    confidence=99)
    source = [dict(text='The', bbox=[10, 20, 30, 32], confidence=99), whole, fragment,
              dict(text='stays', bbox=[120, 20, 150, 32], confidence=99)]
    witness = pa.CropWitness([dict(w, confidence=60 if evidence == 'unconfirmed' else 99)
                              for w in source if w is not fragment])
    result = pa.adjudicate(['The', 'garden', 'stays'], source, witness)
    assert result['tokens'] == (['The', 'garden', 'stays'] if evidence == 'complete'
                                else ['The', 'garden', fragment_text, 'stays'])
    if evidence == 'complete':
        assert not result['marked'], 'A proved whole occurrence cannot acquire its own suffix as a new word'
