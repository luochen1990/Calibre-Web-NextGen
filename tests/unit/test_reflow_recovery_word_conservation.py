"""Synthetic source words and geometry for omitted citation-strip ownership."""
from types import SimpleNamespace
import pymupdf
import pytest
from cps.services.reflow import vision_pipeline as vp

pytestmark=pytest.mark.unit


def test_short_printed_run_between_note_units_is_not_silently_lost():
    text='Collected examples volume three;'
    words=[dict(text=t,bbox=[30+i*40,100,65+i*40,110])for i,t in enumerate(text.split())]
    display=SimpleNamespace(rect=pymupdf.Rect(0,0,300,400))
    def block(ident,text,top,bottom):
        return dict(id=ident,text=text,role='footnote',marker='7' if ident=='left' else '8',
            bbox=[80,top/400*1000,900,bottom/400*1000],low_confidence=[],
            continues_previous=False,continues_next=False)
    answer=dict(blocks=[block('left','A printed note ends here.',70,85),
                        block('right','Another printed note begins here.',130,145)])
    regions={'left':[[24,70,270,85]],'right':[[24,130,270,145]]}
    recovered,ids=vp.cover_missing_lines(answer,words,words,display,regions,{})
    assert len(ids)==1
    inserted=next(b for b in recovered['blocks']if b['id'] in ids)
    assert inserted['text']==text
    assert regions[inserted['id']]==[[30,100,185,110]]
    assert {_ for _ in regions['_instances'].values()}=={inserted['id']}
