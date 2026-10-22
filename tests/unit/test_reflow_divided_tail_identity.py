"""Synthetic divided compounds require evidence of both printed pieces."""
import pytest
from cps.services.reflow import pixel_adjudication as pa

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('prefix',[False,True])
def test_cropped_compound_tail_cannot_certify_a_different_complete_word(prefix):
    source=[dict(text='case',bbox=[20,35,45,45],confidence=99)]
    witness=pa.CropWitness([dict(source[0],text='ease')]);witness.model_reading=True
    witness.word_readings=lambda word:[[dict(word,text='ease',confidence=99)]]*2
    witness.neighbor_occurrences=([dict(text='book-',bbox=[140,20,170,30],confidence=99)]if prefix else [])
    result=pa.adjudicate(['book-case'],source,witness)
    assert result['tokens']==['book-case'],'A tail-only view cannot replace the full divided compound'
    note=result['word_notes'][0]
    assert note['status']=='UNRESOLVED'and note['source_image_only']
    assert note['method']=='unproved_complete_divided_word'
    assert note['unplaced_prefix']is not prefix
    if prefix:assert note['source_parts']==[[140,20,170,30],[20,35,45,45]]


def test_complete_literal_word_keeps_its_independent_correcting_evidence():
    source=[dict(text='case',bbox=[20,35,45,45],confidence=99)]
    witness=pa.CropWitness([dict(source[0],text='ease')])
    result=pa.adjudicate(['case'],source,witness)
    assert result['tokens']==['case']and not result['marked']
