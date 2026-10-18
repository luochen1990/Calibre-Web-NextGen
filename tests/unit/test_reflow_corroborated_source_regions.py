"""Synthetic words and coordinates for source-ownership placement controls."""
from types import SimpleNamespace

import pymupdf
import pytest

from cps.services.reflow import vision_pipeline as vp

pytestmark = pytest.mark.unit


def word(text, x, y, width=28):
    return dict(text=text, bbox=[x, y, x+width, y+8])


def bind(text, role, hint, optical):
    block = dict(id='sample', text=text, role=role, bbox=hint,
                 marker='7' if role == 'footnote' else '', low_confidence=[],
                 continues_previous=False, continues_next=False)
    regions = {}
    display = SimpleNamespace(rect=pymupdf.Rect(0, 0, 500, 720), pno=0)
    answer, _ = vp.bind_geometry(dict(blocks=[block]), optical, display, regions, {})
    assert answer['blocks'][0]['text'] == text
    return regions.get('sample')


def test_displaced_note_hint_cannot_be_published_as_source_ownership():
    text = 'The narrow bridge crosses a quiet stream beside the mill.'
    optical = [word(t, 40+i*35, 580) for i, t in enumerate(text.split())]
    regions = bind(text, 'footnote', [70, 900, 900, 925], optical)
    assert regions is not None
    assert min(r[1] for r in regions) == 580
    assert max(r[3] for r in regions) == 588


def test_short_accent_variant_uses_printed_row_instead_of_following_row():
    optical = [word('Cafeteria.', 40, 205, 45), word('See', 90, 205, 18),
               word('Kitchen.', 113, 205, 40), word('A', 40, 218),
               word('different', 75, 218), word('entry.', 110, 218)]
    regions = bind('Caféteria. See Kitchen.', 'body', [75, 304, 310, 322], optical)
    assert regions is not None
    assert regions == [[40, 205, 153, 213]]


def test_distant_common_word_cannot_stretch_source_region_over_other_entries():
    optical = [word('on', 40, 172), word('other', 75, 172), word('details', 110, 172)]
    optical += [word(t, 40+i*36, 375) for i, t in enumerate(
        'on cafe equipment 12-14, 17, 19'.split())]
    regions = bind('on café equipment 12–14, 17, 19', 'body',
                   [70, 550, 650, 580], optical)
    assert regions == [[40, 375, 248, 383]]


def test_equal_repeated_readings_do_not_identify_one_source_occurrence():
    from cps.services.reflow.source_geometry import unique_anchors
    text = 'The clear narrow bridge crosses a quiet stream.'
    optical = [word(t, 40+i*35, y) for y in (100, 300)
               for i, t in enumerate(text.split())]
    assert unique_anchors(optical, range(len(optical)), text.split()) == []
