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
