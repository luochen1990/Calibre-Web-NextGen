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


def test_same_row_unclaimed_word_between_note_anchors_stays_in_its_source_crop():
    optical=[dict(text='7',bbox=[40,520,45,527])]
    text='The narrow bridge remains Brindle open beside the quiet mill.'
    optical += [dict(text=t,bbox=[60+i*35,520,90+i*35,528])for i,t in enumerate(text.split())]
    answer={'blocks':[dict(note('gap','7'),text=text.replace('Brindle','Brindlc'))]}
    regions={}
    display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    vp.bind_geometry(answer,optical,display,regions,{})
    assert len(regions['gap'])==1
    missing=next(w for w in optical if w['text']=='Brindle')
    assert vp.owned_words([missing],regions['gap'][0])==[missing]
    assert regions['_instances'][vp._source_instance(missing)]=='gap'


def test_unmatched_interior_note_row_is_owned_once_between_its_proved_rows():
    optical=[dict(text='7',bbox=[40,520,45,527])]
    rows=['The narrow bridge remains open.', 'Brindle carried another sample.',
          'The quiet mill keeps working.']
    for row, text in enumerate(rows):
        optical += [dict(text=t,bbox=[60+i*42,520+row*11,98+i*42,528+row*11])
                    for i,t in enumerate(text.split())]
    text=' '.join(rows).replace(rows[1], 'Brindlc carricd anothcr samplc.')
    answer={'blocks':[dict(note('middle','7'),text=text)]}
    regions={}
    display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    vp.bind_geometry(answer,optical,display,regions,{})
    middle=[w for w in optical if w['bbox'][1]==531]
    assert all(any(vp.owned_words([w],r) for r in regions['middle']) for w in middle)
    assert all(regions['_instances'][vp._source_instance(w)]=='middle' for w in middle)
