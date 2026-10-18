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


def test_compact_unique_pixel_proof_overrides_more_broad_fuzzy_anchors(monkeypatch):
    text = 'The bright bridge remains open beside seven mills.'
    optical = [word('clear' if t=='bright' else t, 40+i*35, 200)
               for i, t in enumerate(text.split())]
    optical += [word('The', 40, 400)]
    # A permissive matcher can count more tokens by borrowing a remote copy.
    # A coherent exact occurrence must dominate that unrelated physical union.
    original = vp._geometry_anchors
    def broad(*args, **kwargs):
        if args[0] is not optical:return original(*args, **kwargs)
        return [(len(optical)-1, 0)]+[(i, i) for i in range(1, len(text.split()))]
    monkeypatch.setattr(vp, '_geometry_anchors', broad)
    regions = bind(text, 'body', [70, 260, 650, 570], optical)
    assert regions == [[40, 200, 313, 208]]


def test_native_sequence_can_locate_wrapped_numeric_ink_without_certifying_digits(monkeypatch):
    text='near mills 121 232 343 454 565–66 676–77 787–88'
    native=[word(t,40+i*42,200,38) for i,t in enumerate(
        'near mills 121 232 343 454 565-66 676-'.split())]
    native += [word('77',60,211,20),word('787-88',85,211,38)]
    optical=[dict(w) for w in native]
    for i,t in [(3,'$32'),(4,'$43'),(5,'$54'),(6,'65-66'),(7,'$76-')]:
        optical[i]=dict(optical[i],text=t)
    monkeypatch.setattr(vp,'layer_words',lambda *args:native)
    block=dict(id='wrapped',text=text,role='body',bbox=[70,300,650,330],marker='',
               low_confidence=[],continues_previous=False,continues_next=False)
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0,doc=object())
    bound,_=vp.bind_geometry({'blocks':[block]},optical,display,regions,{})
    assert bound['blocks'][0]['text']==text
    assert min(r[1] for r in regions['wrapped'])==200
    assert max(r[3] for r in regions['wrapped'])==219
    assert all(regions['_instances'][vp._source_instance(w)]=='wrapped' for w in optical)


def test_joint_views_place_one_interior_variant_but_do_not_correct_it(monkeypatch):
    text='The bright renovation 12–13'
    native=[word(t,40+i*50,200,46) for i,t in enumerate('The bright renovations 12-13'.split())]
    optical=[dict(w) for w in native]
    monkeypatch.setattr(vp,'layer_words',lambda *args:native)
    block=dict(id='variant',text=text,role='body',bbox=[70,300,650,330],marker='',
               low_confidence=[],continues_previous=False,continues_next=False)
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0,doc=object())
    bound,_=vp.bind_geometry({'blocks':[block]},optical,display,regions,{})
    assert bound['blocks'][0]['text']==text
    assert regions['variant']==[[40,200,236,208]]


def test_joint_placement_needs_visible_ink_and_unique_physical_occurrence():
    from cps.services.reflow.source_geometry import native_pixel_anchors
    native=[word(t,40+i*50,200,46) for i,t in enumerate('The bright renovations 12-13'.split())]
    candidate='The bright renovation 12–13'.split()
    assert native_pixel_anchors(native,[],[],candidate)==[]
    repeated=native+[dict(w,bbox=[w['bbox'][0],400,w['bbox'][2],408]) for w in native]
    assert native_pixel_anchors(repeated,repeated,range(len(repeated)),candidate)==[]


def test_compact_occurrence_wins_over_shared_anchors_with_extra_source_words():
    from cps.services.reflow.source_geometry import unique_anchors
    optical=[word(t,40+i*35,200)for i,t in enumerate('near other mills 121 232 343'.split())]
    optical += [word(t,40+i*35,212)for i,t in enumerate('near Brindle 454 565 676 787 898'.split())]
    candidate='near Brindlc 454 565 676 787 898'.split()
    proof=unique_anchors(optical,range(len(optical)),candidate)
    assert proof
    assert all(optical[i]['bbox'][1]==212 for i,j in proof)


def test_native_wrapped_placement_ignores_an_interleaved_other_column():
    from cps.services.reflow.source_geometry import native_pixel_anchors
    native=[word(t,100+i*35,200,30)for i,t in enumerate('near mills 121 232 343 454 565-66 676-'.split())]
    native += [word('77',120,211,20),word('787-88',145,211,30),word('Elsewhere',0,207,28)]
    optical=[dict(w)for w in native]
    for i,t in [(3,'$32'),(4,'$43'),(5,'$54'),(6,'65-66'),(7,'$76-')]:optical[i]=dict(optical[i],text=t)
    proof=native_pixel_anchors(native,optical,range(len(optical)),
        'near mills 121 232 343 454 565–66 676–77 787–88'.split())
    assert {j for i,j in proof}==set(range(9))
    assert all(i!=10 for i,j in proof)


def test_running_head_cannot_relocate_to_matching_body_words():
    text='The clear sample remains beside Brindle.'
    optical=[word('D4MAGED',40,12),word('HEADER',90,12)]
    optical.extend(word(t,40+i*55,200)for i,t in enumerate(text.split()))
    header=dict(id='head',text=text,role='running_head',bbox=[0,0,1000,70],marker='',
                low_confidence=[],continues_previous=False,continues_next=False)
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    answer,_=vp.bind_geometry({'blocks':[header]},optical,display,regions,{})
    assert answer['blocks'][0]['bbox']==header['bbox']
    assert not any(vp.owned_words([w],r)for w in optical if w['bbox'][1]==200
                   for r in regions.get('head',[]))
