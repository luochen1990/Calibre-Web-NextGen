"""Synthetic vector and scanned line previews preserve gray ink and descent."""
import io
import pymupdf
import pytest
from PIL import Image
from cps.services.reflow import inline_print
from cps.services.reflow.source_display import SourceDisplay

pytestmark=pytest.mark.unit


@pytest.mark.parametrize('scanned', [False, True])
def test_inline_word_uses_printed_line_scale_instead_of_paper_box(scanned):
    with pymupdf.open() as doc, pymupdf.open() as scan:
        p=doc.new_page(width=300,height=150)
        p.insert_text((30,70),'The garden grows green.',fontsize=16)
        word=next(w for w in p.get_text('words')if w[4]=='garden')
        if scanned:
            page=scan.new_page(width=300,height=150)
            page.insert_image(page.rect,pixmap=p.get_pixmap(matrix=pymupdf.Matrix(3,3)))
        display=SourceDisplay(scan if scanned else doc,0)
        value=inline_print.preview(display,list(word[:4]))
    assert value is not None
    data,metric=value
    image=Image.open(io.BytesIO(data)).convert('L')
    assert data.startswith(b'\x89PNG')
    assert metric['vertical_align_em']<0
    assert metric['height_em']>1
    assert len(set(image.tobytes()))>2,'Anti-aliased strokes must not be thresholded away'
    assert metric['bbox'][1]<metric['baseline']<metric['bbox'][3]


def test_multiline_source_box_cannot_supply_single_line_preview_metrics():
    with pymupdf.open() as doc:
        page=doc.new_page(width=350,height=180)
        page.insert_text((30,60),'The garden grows green.',fontsize=16)
        page.insert_text((30,90),'The clear sample remains readable.',fontsize=16)
        words=page.get_text('words')
        box=[min(w[0] for w in words),min(w[1] for w in words),
             max(w[2] for w in words),max(w[3] for w in words)]
        assert inline_print.preview(SourceDisplay(doc,0),box) is None


@pytest.mark.parametrize('case',['refused','dense','sparse','sparse_split_territories'])
def test_page_publication_uses_a_unit_print_instead_of_a_shrunken_inspection_asset(tmp_path,monkeypatch,case):
    from cps.services.reflow import vision_pipeline as vp,pixel_adjudication as pa
    from tests.unit.test_reflow_vision_pilot import answer,block
    printed='The clear garden grows beside the quiet green path in summer.'
    with pymupdf.open()as drawn,pymupdf.open()as scan:
        source=drawn.new_page(width=420,height=160)
        source.insert_text((20,60),printed,fontsize=12)
        words=[dict(text=w[4],bbox=list(w[:4]),confidence=99)for w in source.get_text('words')]
        page=scan.new_page(width=420,height=160);page.insert_image(page.rect,pixmap=source.get_pixmap())
        monkeypatch.setattr(pa,'read_crop',lambda *a,**k:pa.CropWitness(words))
        original=pa.adjudicate
        def uncertain(*a,**kw):
            result=original(*a,**kw)
            indices=range(len(result['tokens']))if case=='dense'else[2]
            for i in indices:
                result['marked'].add(i)
                result.setdefault('word_notes',{})[i]=dict(status='UNRESOLVED',
                    source=[result['tokens'][i]],model=[result['tokens'][i]],chosen=[result['tokens'][i]],
                    method='unproved_example',source_range=[i,i+1],model_range=[i,i+1])
            return result
        monkeypatch.setattr(pa,'adjudicate',uncertain)
        if case=='refused':monkeypatch.setattr(inline_print,'preview',lambda *a:None)
        class Package:
            def __init__(self):self.images={}
            def image(self,name,data):self.images[name]=data
        package=Package();assets={}
        territories=[[15,40,410,75]]
        if case=='sparse_split_territories':
            split=(words[5]['bbox'][2]+words[6]['bbox'][0])/2
            territories=[[15,40,split,75],[split,40,410,75]]
        compiled,receipt=vp.compile_page(answer(block(printed)),words,SourceDisplay(scan,0),0,
            package,assets,pixel_words=words,regions={'b1':territories},body_owners={},
            note_context={'cache':tmp_path/'cache','labels':{},'catalogue':{},'refs':{}})
    from lxml import etree
    root=etree.fromstring(('<div xmlns:epub="http://www.idpf.org/2007/ops">'+''.join(compiled['body'])+'</div>').encode())
    flags=root.xpath('.//span[@class="reflow-flagged-reading"]')
    assert flags,'The plausible reading must stay visibly flagged'
    assert receipt['blocks'][0]['published_tokens']==printed.split()
    if case in ('refused','dense'):
        assert not root.xpath('.//span[@class="reflow-flagged-reading"]//img'), 'No broad inspection image may impersonate an inline word'
        unit=root.xpath('.//div[@class="reflow-unit-print"]')
        assert len(unit)==1,'Uncertain unit evidence is displayed once after the paragraph'
        assert root.index(unit[0])>0
        assert len(root.xpath('.//sup[text()="†"]'))==1
        image=unit[0].xpath('.//img')[0]
        assert 'height:1.4em'not in image.get('style','')
        assert image.get('src')in package.images
        if case=='dense':assert not any(v.get('kind')=='inline_print_preview'for v in assets.values()), 'Unused word previews must not fill the archive after choosing a unit print'
    else:
        assert not root.xpath('.//div[@class="reflow-unit-print"]')
        assert len(root.xpath('.//span[@class="reflow-flagged-reading"]//img'))==1
        assert all(assets[e.get('src')]['kind']=='inline_print_preview'
                   for e in root.xpath('.//span[@class="reflow-flagged-reading"]//img'))


def test_disconnected_capital_crossbar_inside_word_box_survives_line_metrics():
    with pymupdf.open()as doc:
        page=doc.new_page(width=240,height=150)
        # Disconnected top ink is still part of the bounded printed capital.
        page.draw_rect(pymupdf.Rect(40,75,54,76),color=None,fill=(0,0,0))
        page.draw_rect(pymupdf.Rect(46,80,48,95),color=None,fill=(0,0,0))
        for x in range(70,150,10):
            page.draw_rect(pymupdf.Rect(x,84,x+6,95),color=None,fill=(0,0,0))
        value=inline_print.preview(SourceDisplay(doc,0),[39,74,55,96])
    assert value is not None
    image=Image.open(io.BytesIO(value[0])).convert('L')
    components=list(inline_print._components((v<128 for v in image.tobytes()),image.width,image.height))
    bounds=[(max(q%image.width for q in c)-min(q%image.width for q in c)+1,
             max(q//image.width for q in c)-min(q//image.width for q in c)+1)for c in components]
    assert any(width>=5*height for width,height in bounds), 'The disconnected printed crossbar must not be filtered away'


def test_faded_print_has_near_black_stroke_cores_without_erasing_gray_edges():
    with pymupdf.open()as doc:
        pg=doc.new_page(width=300,height=150)
        pg.insert_text((30,70),'The garden grows green.',fontsize=16,color=(.5,.5,.5))
        word=next(w for w in pg.get_text('words')if w[4]=='garden')
        value=inline_print.preview(SourceDisplay(doc,0),list(word[:4]))
    assert value is not None
    image=Image.open(io.BytesIO(value[0])).convert('L')
    ink=sorted(v for v in image.tobytes()if v<200)
    assert ink and ink[len(ink)//5]<60,'Faded core strokes need contrast that survives monochrome dithering'
    assert len(set(image.tobytes()))>2,'Gray edge samples remain instead of a binary redraw'
