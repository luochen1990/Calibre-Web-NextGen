"""Synthetic late alphabet candidates preserve the printed accent question."""
import pytest
from cps.services.reflow import pixel_adjudication as pa
from tests.unit.test_reflow_pixel_adjudication import words

pytestmark=pytest.mark.unit


def test_late_alternate_pair_cannot_certify_dropping_a_reader_macron():
    source=words('Read this clear samplum beside the open window.')
    witness=pa.CropWitness(source);witness.alternates=lambda:[source,source]
    model='Read this clear samplūm beside the open window.'.split()
    result=pa.adjudicate(model,source,witness)
    assert result['tokens']==model
    note=result['word_notes'][3]
    assert note['status']=='UNRESOLVED'and note.get('source_image_only')
    assert note['method']=='unproved_diacritic_identity'
