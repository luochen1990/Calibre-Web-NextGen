"""Synthetic Roman reference ink must not be certified as a decimal digit."""
import pytest
from cps.services.reflow import pixel_adjudication as pa
from tests.unit.test_reflow_pixel_adjudication import words

pytestmark=pytest.mark.unit


def test_repeated_ocr_digit_cannot_certify_replacing_a_roman_reference():
    source=words('Read section 1.7 beside the sample.')
    witness=pa.CropWitness(source);witness.word_readings=lambda w:[[dict(w)]]*2
    result=pa.adjudicate('Read section I.7 beside the sample.'.split(),source,witness)
    note=result['word_notes'][2]
    assert note['status']=='UNRESOLVED'and note.get('source_image_only')
    assert note['method']=='unproved_reference_numeral_identity'


@pytest.mark.parametrize('context',['See','scale'])
def test_unsettled_local_reference_is_print_but_ordinary_decimal_is_live(context):
    source=words(context+' 1.7 units')
    result=dict(tokens=[w['text']for w in source],word_notes={},marked=set())
    pa.audit_local_numeric_suffixes(result,{i:w['bbox']for i,w in enumerate(source)},pa.CropWitness(source))
    assert bool(result['word_notes'].get(1,{}).get('source_image_only'))is(context=='See')
    assert result['tokens']==[w['text']for w in source],'The source image preserves the numeral; no replacement I is guessed'
