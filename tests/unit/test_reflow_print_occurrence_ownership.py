"""Synthetic printed rows retain one owner across different recognition boxes."""
import pymupdf
import pytest

from cps.services.reflow import vision_pipeline
from cps.services.reflow.source_display import SourceDisplay
from tests.unit.test_reflow_vision_pilot import answer, block

pytestmark = pytest.mark.unit


class Package:
    def disclosure_image(self, *args):
        pass


def test_native_row_is_not_repeated_through_neighbor_optical_territory(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=500, height=250)
        page.insert_text((30, 60), 'Entry for the northern road 12', fontsize=12)
        page.insert_text((30, 90), 'Entry for the western road 15', fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        optical = [dict(w, bbox=[x + .4 for x in w['bbox']]) for w in words]
        regions = {'b1': [[25, 40, 300, 100]], 'b2': [[25, 75, 300, 100]],
                   '_instances': {vision_pipeline._source_instance(w): 'b1' if i < 6 else 'b2'
                                  for i, w in enumerate(optical)}}
        reading = answer(dict(block('Entry for the northern road 12'), id='b1'),
                         dict(block('Entry for the western road 15'), id='b2'))
        _, record = vision_pipeline.compile_page(
            reading, words, SourceDisplay(doc, 0), 0, Package(), {}, pixel_words=optical,
            regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [
        'Entry for the northern road 12'.split(), 'Entry for the western road 15'.split()]


def test_two_distinct_printed_occurrences_keep_both_readings(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=500, height=250)
        text = 'The repeated line remains clearly printed.'
        page.insert_text((30, 60), text, fontsize=12)
        page.insert_text((30, 100), text, fontsize=12)
        words = [dict(text=w[4], bbox=list(w[:4]), confidence=99) for w in page.get_text('words')]
        regions = {'b1': [[25, 40, 350, 65]], 'b2': [[25, 80, 350, 105]]}
        reading = answer(block(text, ident='b1'), block(text, ident='b2'))
        _, record = vision_pipeline.compile_page(
            reading, words, SourceDisplay(doc, 0), 0, Package(), {}, pixel_words=words,
            regions=regions, body_owners={},
            note_context={'cache': tmp_path/'cache', 'labels': {}, 'catalogue': {}, 'refs': {}})
    assert [b['published_tokens'] for b in record['blocks']] == [text.split(), text.split()]
