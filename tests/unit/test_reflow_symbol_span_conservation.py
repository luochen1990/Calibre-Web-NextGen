"""Synthetic mathematical ink must not become alphabetic OCR guesses."""
import pytest
from cps.services.reflow import pixel_adjudication as pa

pytestmark=pytest.mark.unit


def test_multiword_symbol_edit_cannot_split_into_confident_letter_deletions():
    before='Please compare the symbols T, D. Q, c. beside this clear example.'
    model='Please compare the symbols ∂ ∑ ∞ ∫ beside this clear example.'
    source=[dict(text=t,bbox=[i*25,20,i*25+22,32],confidence=99)
            for i,t in enumerate(before.split())]
    witness=pa.CropWitness(source)
    witness.model_reading=True
    result=pa.adjudicate(model.split(),source,witness)
    assert result['tokens'][:4]==before.split()[:4]
    assert result['tokens'][-4:]==before.split()[-4:]
    for i in range(4,8):
        note=result['word_notes'].get(i,{})
        assert note.get('source_image_only'), 'Alphabetic recognizer agreement does not identify mathematical ink'
        assert note['status']=='UNRESOLVED'
        assert note['source_bbox']==source[i]['bbox']
    assert set(result['marked'])==set(range(4,8))


def test_literal_latin_letters_with_matching_readings_stay_live():
    text='Please compare the labels T D Q c beside this clear example.'
    source=[dict(text=t,bbox=[i*25,20,i*25+22,32],confidence=99)
            for i,t in enumerate(text.split())]
    witness=pa.CropWitness(source);witness.model_reading=True
    result=pa.adjudicate(text.split(),source,witness)
    assert result['tokens']==text.split()
    assert not result['marked']
