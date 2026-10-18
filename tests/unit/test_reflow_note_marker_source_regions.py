"""Synthetic repeated note prose, labels and geometry written for these tests."""
from types import SimpleNamespace

import pymupdf
import pytest

from cps.services.reflow import vision_pipeline as vp

pytestmark = pytest.mark.unit


def note(ident, marker):
    return dict(id=ident, role='footnote', marker=marker,
                text='The narrow bridge remains open beside the quiet mill.',
                bbox=[170, 860, 900, 885], low_confidence=[],
                continues_previous=False, continues_next=False)


def words(marker, y):
    result = [dict(text=marker, bbox=[40, y, 45, y+7])]
    for i, token in enumerate(note('', '')['text'].split()):
        result.append(dict(text=token, bbox=[60+i*35, y, 90+i*35, y+8]))
    return result


def test_repeated_note_text_binds_to_its_own_printed_label():
    answer = dict(blocks=[note('first', '7'), note('second', '8')])
    optical = words('7', 520) + words('8', 540)
    regions = {}
    display = SimpleNamespace(rect=pymupdf.Rect(0, 0, 500, 720), pno=0)
    bound, _ = vp.bind_geometry(answer, optical, display, regions, {})
    assert [b['text'] for b in bound['blocks']] == [b['text'] for b in answer['blocks']]
    assert regions['first'] == [[40, 520, 370, 528]]
    assert regions['second'] == [[40, 540, 370, 548]]
    covered = [vp.owned_words(optical, region) for values in regions.values()
               if isinstance(values, list) for region in values]
    assert sum(len(row) for row in covered) == len(optical)


def test_unproved_label_cannot_select_a_nearby_repeated_note():
    from cps.services.reflow.source_geometry import labelled_note_anchors
    optical = words('7', 520) + words('8', 540)
    assert labelled_note_anchors(optical, range(len(optical)),
                                 note('', '')['text'].split(), '9') is None


def test_reserved_following_label_still_bounds_the_original_note():
    from cps.services.reflow.source_geometry import labelled_note_anchors
    first = words('7', 520)
    next_note = words('8', 540)
    later = words('?', 560)
    optical = first + next_note + later
    available = list(range(len(first))) + list(range(len(first)+len(next_note), len(optical)))
    proof = labelled_note_anchors(optical, available, note('', '')['text'].split(), '7')
    assert proof is not None
    assert proof['marker_bbox'] == [40, 520, 45, 527]
    assert {optical[i]['bbox'][1] for i, _ in proof['anchors']} == {520}


def test_printed_label_and_four_words_locate_a_damaged_opening_without_certifying_it():
    answer=dict(blocks=[dict(note('short','12'),text='Brindle brought the clear sample.')])
    optical=[dict(text='12',bbox=[40,540,49,547])]
    optical += [dict(text=t,bbox=[60+i*40,540,96+i*40,548])
                for i,t in enumerate('Brindlc brought the clear sample.'.split())]
    regions={}
    display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    bound,_=vp.bind_geometry(answer,optical,display,regions,{})
    assert bound['blocks'][0]['text']=='Brindle brought the clear sample.'
    assert regions['short']==[[40,540,256,548]]
