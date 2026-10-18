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


def test_note_continuation_can_return_to_the_printed_label_margin():
    from cps.services.reflow.source_geometry import labelled_note_anchors
    rows=['Please carry the clear', 'sample beside the quiet', 'window today.']
    optical=[dict(text='7',bbox=[40,520,45,527])]
    for row,text in enumerate(rows):
        left=60 if row==0 else 40
        optical.extend(dict(text=t,bbox=[left+i*40,520+row*11,left+i*40+35,528+row*11])
                       for i,t in enumerate(text.split()))
    following=len(optical)
    optical.extend([dict(text='8',bbox=[40,560,45,567]),dict(text='Other',bbox=[60,560,95,568])])
    proof=labelled_note_anchors(optical,range(len(optical)),' '.join(rows).split(),'7')
    assert proof is not None
    assert [optical[i]['text']for i,j in proof['anchors']]==' '.join(rows).split()
    assert all(i<following for i,j in proof['anchors'])


def test_local_printed_note_opening_cannot_relocate_to_a_distant_equal_label():
    distant=words('7',520)
    local=[dict(text='7',bbox=[40,620,45,627])]
    local += [dict(text=t,bbox=[60+i*35,620,90+i*35,628])for i,t in enumerate(
        'Other authored details stay distinct beside this sample.'.split())]
    block=dict(note('local','7'),bbox=[60,850,900,900])
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    bound,_=vp.bind_geometry({'blocks':[block]},distant+local,display,regions,{})
    assert not regions.get('local') or min(r[1]for r in regions['local'])>=612
    assert bound['blocks'][0]['bbox'][1]>=850,'a matching distant label cannot overrule the printed local opening'


def test_differently_recognized_note_label_retains_an_explicit_owned_occurrence():
    block=dict(note('opening','†'),bbox=[60,850,900,900])
    optical=words('+',620)
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    vp.bind_geometry({'blocks':[block]},optical,display,regions,{})
    proof=regions.get('_note_openings',{}).get('opening',{})
    assert proof.get('source_bbox')==[40,620,45,627]
    assert proof['observed_text']=='+' and proof['expected_label']=='†'
    assert proof['identity_status']=='UNPROVED','geometry locates a label without certifying its identity'

    remaining=vp._note_lexical_evidence(optical,proof)
    assert remaining==optical[1:]
    remote=dict(optical[0],bbox=[40,640,45,647])
    assert vp._note_lexical_evidence([remote],proof)==[remote]


def test_complete_two_column_note_is_not_replaced_by_a_partial_labelled_prefix():
    rows=['The narrow bridge remains','beside the quiet mill.','A careful worker waits','Another sample arrives','before evening rain.']
    optical=[dict(text='*',bbox=[24,500,29,507])]
    for number,text in enumerate(rows):
        x,y=(40,500+12*number) if number<3 else (210,500+12*(number-3))
        optical += [dict(text=t,bbox=[x+32*i,y,x+32*i+28,y+8])for i,t in enumerate(text.split())]
    block=dict(note('continued','*'),text=' '.join(rows),bbox=[40,680,900,770])
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    vp.bind_geometry({'blocks':[block]},optical,display,regions,{})
    actual=vp._owned_block_words(block,{'blocks':[block]},optical,display,regions,ordered=True)
    assert [w['text']for w in vp._note_lexical_evidence(actual,regions.get('_note_openings',{}).get('continued'))]==block['text'].split()


def test_complete_note_territory_does_not_prove_an_interior_label_partition():
    optical=[dict(text=t,bbox=[40+32*i,500,68+32*i,508])for i,t in enumerate('A careful worker waits.'.split())]
    optical += [dict(text='+',bbox=[24,520,29,527])]
    optical += [dict(text=t,bbox=[40+32*i,520,68+32*i,528])for i,t in enumerate('Another printed citation follows.'.split())]
    block=dict(note('merged','†'),text='A careful worker waits. Another printed citation follows.',bbox=[40,680,900,770])
    regions={'merged':[[24,500,180,528]]};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    assert vp._unpartitioned_note_openings({'blocks':[block]},optical,display,regions,{})==[
        dict(block='merged',source_bbox=[24,520,29,527])]
    initial=dict(text='+',bbox=[24,500,29,507])
    control=[initial]+[w for w in optical if w['text']!='+']
    regions['_note_openings']={'merged':dict(observed_text='+',source_bbox=initial['bbox'])}
    assert vp._unpartitioned_note_openings({'blocks':[block]},control,display,regions,{})==[]


def test_complete_note_placement_does_not_choose_between_repeated_occurrences():
    text='The narrow bridge remains beside the quiet mill.'
    optical=[dict(text=t,bbox=[40+32*i,y,68+32*i,y+8])for y in (500,525)for i,t in enumerate(text.split())]
    assert vp._note_pixel_anchors(optical,range(len(optical)),text.casefold().replace('.','').split(),
        [20,480,400,550],exact_complete=True)==[]


def test_complete_nearby_note_keeps_the_opening_above_a_tight_tail_hint():
    rows=['Another printed source note begins beside the window.',
          'Its continuation retains several ordinary printed details.',
          'The careful worker records the remaining samples.']
    optical=[dict(text='7',bbox=[24,500,29,507])]
    for row,text in enumerate(rows):
        optical += [dict(text=t,bbox=[40+i*35,500+row*14,71+i*35,508+row*14])
                    for i,t in enumerate(text.split())]
    block=dict(note('nearby','7'),text=' '.join(rows),bbox=[40,715,900,765])
    regions={};display=SimpleNamespace(rect=pymupdf.Rect(0,0,500,720),pno=0)
    proof=vp._note_pixel_anchors(optical,range(len(optical)),
        block['text'].casefold().replace('.','').split(),vp.block_rect(block,display),exact_complete=True)
    assert {j for i,j in proof}==set(range(len(block['text'].split())))
    vp.bind_geometry({'blocks':[block]},optical,display,regions,{})
    assert all(regions.get('_instances',{}).get(vp._source_instance(w))=='nearby' for w in optical)
