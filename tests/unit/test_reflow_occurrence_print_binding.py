"""Synthetic sequence ownership keeps a plausible reading's print honest."""
import io
import pymupdf
import pytest
from lxml import etree
from PIL import Image
from cps.services.reflow import inline_print,vision_pipeline as vp,reading_tiers
from cps.services.reflow.source_display import SourceDisplay

pytestmark=pytest.mark.unit

class Package:
    def __init__(self):self.images={}
    def image(self,name,data):
        assert name not in self.images
        self.images[name]=data


def published(display,words,reading,wrong,*,heading=False):
    package=Package();images={'images/old.jpg':dict(kind='vision_disclosure',bbox=wrong)};extras=[]
    text=reading.replace('garden',reading_tiers.flagged_html('garden','images/old.jpg','test-print'))
    text+='<a id="r-test-print" class="source-disclosure-ref" epub:type="noteref" href="#test-print"><sup>†</sup></a>'
    body,supplement=vp._present_flagged_prints(text,display,0,'sample',[wrong],package,images,extras,
        evidence_words=words,heading=heading)
    return body,supplement,package,images,extras


def test_shared_evidence_uses_its_unique_reading_not_a_wrong_row(monkeypatch):
    with pymupdf.open()as doc:
        p=doc.new_page(width=400,height=230)
        p.insert_text((25,55),'The blue garden stays green beside the quiet path.',fontsize=12)
        p.insert_text((25,155),'The red tractor waits beside the old barn.',fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in p.get_text('words')]
        monkeypatch.setattr(inline_print,'preview',lambda *a:None)
        _,supplement,package,images,_=published(SourceDisplay(doc,0),words,
            'The blue garden stays green beside the quiet path.',[20,140,340,160])
        asset=next(v for v in images.values()if v.get('kind')=='flagged_unit_preview')
        assert asset['bbox'][3]<80
        assert asset['placement_method']=='unique_sequence_source_context'
        root=etree.fromstring('<div xmlns:epub="http://www.idpf.org/2007/ops">'+supplement+'</div>')
        assert len(root.xpath('.//img'))==1


def test_ambiguous_reading_gets_complete_page_context_not_a_guessed_word(monkeypatch):
    with pymupdf.open()as doc:
        p=doc.new_page(width=400,height=230)
        for y in(55,155):p.insert_text((25,y),'blue garden',fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in p.get_text('words')]
        _,_,_,images,_=published(SourceDisplay(doc,0),words,'blue garden',[20,140,140,160])
        asset=next(v for v in images.values()if v.get('kind')=='flagged_unit_preview')
        assert asset['bbox']==list(p.rect)
        assert asset['unplaced_context']


def test_wrong_inline_preview_is_rebound_before_it_can_stay_inline():
    with pymupdf.open()as doc:
        p=doc.new_page(width=400,height=230)
        reading='The blue garden stays green beside the quiet path.'
        p.insert_text((25,55),reading,fontsize=12)
        p.insert_text((25,155),'The red tractor waits beside the old barn.',fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in p.get_text('words')]
        package=Package();images={'images/old.png':dict(kind='inline_print_preview',bbox=[20,140,60,160])}
        text=reading.replace('garden',reading_tiers.flagged_html('garden','images/old.png','test-print','height:9em'))
        body,_,=vp._present_flagged_prints(text,SourceDisplay(doc,0),0,'sample',[[20,140,340,160]],package,images,[],evidence_words=words)
        image=etree.fromstring('<div>'+body+'</div>').xpath('.//img')[0]
        assert image.get('src')!='images/old.png'
        assert images[image.get('src')]['bbox'][3]<80
        assert 'height:1.10000em'in image.get('style')


def test_heading_does_not_repeat_its_words_as_an_inline_or_block_print():
    with pymupdf.open()as doc:
        p=doc.new_page(width=400,height=200);p.insert_text((25,60),'The blue garden stays green.',fontsize=14)
        words=[dict(text=w[4],bbox=list(w[:4]))for w in p.get_text('words')]
        body,supplement,_,_,extras=published(SourceDisplay(doc,0),words,'The blue garden stays green.',[20,45,350,70],heading=True)
        assert '<img'not in body+supplement
        assert 'reflow-plausible'in body and extras


def test_blank_and_impossibly_thin_print_cannot_carry_a_multi_letter_reading():
    for draw in('blank','vertical','horizontal'):
        image=Image.new('L',(80,50),255)
        if draw!='blank':
            from PIL import ImageDraw
            box=(38,5,40,40)if draw=='vertical'else(2,25,78,26)
            ImageDraw.Draw(image).rectangle(box,fill=0)
        out=io.BytesIO();image.save(out,format='PNG')
        assert inline_print.readable_preview((out.getvalue(),{}),'garden')is None


def test_compiler_replaces_stale_flag_crop_at_its_own_occurrence(tmp_path,monkeypatch):
    """The publisher supplies real source words to placement, not just hints."""
    from cps.services.reflow import pixel_adjudication as pa
    from tests.unit.test_reflow_vision_pilot import answer,block
    reading='The blue garden stays green beside the quiet path.'
    with pymupdf.open()as drawing,pymupdf.open()as scan:
        p=drawing.new_page(width=400,height=230)
        p.insert_text((25,55),reading,fontsize=12)
        p.insert_text((25,155),'The red tractor waits beside the old barn.',fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in p.get_text('words')]
        page=scan.new_page(width=400,height=230);page.insert_image(page.rect,pixmap=p.get_pixmap())
        monkeypatch.setattr(pa,'read_crop',lambda *a,**k:pa.CropWitness(words))
        original=pa.adjudicate
        def unsettled(*a,**kw):
            result=original(*a,**kw);index=result['tokens'].index('garden')
            result['marked'].add(index);result['boxes'][index]=[20,140,340,160]
            result.setdefault('word_notes',{})[index]=dict(status='UNRESOLVED',
                source=['garden'],model=['garden'],chosen=['garden'],
                method='synthetic_uncertain_word',source_range=[index,index+1],model_range=[index,index+1])
            return result
        monkeypatch.setattr(pa,'adjudicate',unsettled)
        package=Package();assets={}
        compiled,receipt=vp.compile_page(answer(block(reading)),words,SourceDisplay(scan,0),0,
            package,assets,pixel_words=words,regions={'b1':[[20,40,390,70]]},body_owners={},
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    root=etree.fromstring('<div xmlns:epub="http://www.idpf.org/2007/ops">'+''.join(compiled['body']+compiled['asides'])+'</div>')
    flags=root.xpath('.//span[@class="reflow-flagged-reading"]')
    assert len(flags)==1 and ''.join(flags[0].xpath('.//span[@class="reflow-plausible"]')[0].itertext())=='garden'
    shown=flags[0].xpath('.//img')
    assert shown and assets[shown[0].get('src')]['bbox'][3]<80
    link=flags[0].xpath('.//a')[0].get('href')[1:]
    inspected=root.xpath('//*[@id=$target]//img',target=link)
    assert inspected and assets[inspected[0].get('src')]['bbox'][3]<80
    assert all(image.get('src')in package.images for image in root.xpath('.//img'))
    assert receipt['blocks'][0]['published_tokens']==reading.split()
