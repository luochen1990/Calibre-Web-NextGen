"""Publication tiers keep an unproved reading adjacent to its original print.

These choices supply reading assistance, never an identity certificate. Existing
resolved corrections retain their correction notice; undecodable glyphs and
incomplete surviving fragments remain print-only.
"""
import re
import unicodedata
from html import escape

PRINT_ONLY = frozenset({
    'unproved_native_character_map', 'unproved_doubtful_leading_word_ink',
    'incomplete_vertical_word_ink', 'incomplete_numeric_word_ink',
    'unproved_complete_divided_word', 'unproved_complete_word_scope',
})


def typographic_key(value):
    # NFKC expands presentation ligatures but does not strip accents/macrons.
    value = unicodedata.normalize('NFKC', value).replace('\u00ad', '')
    return value.translate(str.maketrans({'’': "'", '‘': "'", 'ʼ': "'",
                                         '“': '"', '”': '"'}))


def equivalent(left, right):
    return typographic_key(left) == typographic_key(right)


def choose(token, decision):
    """Choose assistance from an existing reader, with an explicit provenance.

    Numeric identifiers prefer a source reading when the model dropped a real
    digit. Ordinary disagreements retain the chosen complete first reading;
    a dictionary cannot distinguish two valid words or authorize added letters.
    """
    if decision.get('status') == 'RESOLVED':
        return dict(tier='proved', reading=token, method='existing_resolution')
    method = decision.get('method', '')
    if method in PRINT_ONLY or decision.get('unplaced_prefix'):
        return dict(tier='print', reading=None, method='no_complete_reading')
    source, model = decision.get('source', []), decision.get('model', [])
    if method == 'unproved_mathematical_symbol_identity':
        # A Unicode symbol is a plausible reading. Alphabetic guesses made by
        # both general readers are not evidence of a symbolic font's identity.
        symbolic = next((s for s in source + model
                         if any(unicodedata.category(c).startswith('S') for c in s)), None)
        if symbolic is None:
            return dict(tier='print', reading=None, method='unidentified_symbol')
        return dict(tier='flagged', reading=symbolic, method='reader_unicode_symbol')
    if len(source) == len(model) == 1:
        a, b = source[0], model[0]
        if (any(c.isdigit() for c in a) and any(c.isalpha() for c in a)
                and sum(c.isdigit() for c in a) > sum(c.isdigit() for c in b)
                and len(a) == len(b) + 1
                and any(a[:i] + a[i+1:] == b for i,c in enumerate(a) if c.isdigit())):
            return dict(tier='flagged', reading=a, method='source_complete_numeric_identifier')
    # Private-use, replacement and control glyphs cannot be helpful text. A
    # source-only punctuation fragment is not a plausible whole word either.
    if (not token or not any(c.isalnum() for c in token)
            or any(unicodedata.category(c) in ('Co', 'Cc', 'Cs') or c == '\ufffd' for c in token)
            or token.startswith('<n:')):
        return dict(tier='print', reading=None, method='no_lexical_reading')
    return dict(tier='flagged', reading=token, method='existing_reader_hypothesis')


def prepare(result):
    """Attach publication decisions without erasing the original adjudication."""
    flags = {}
    for index in sorted(result['marked']):
        if result['tokens'][index].startswith('<n:'):
            continue
        old = result.get('word_notes', {}).get(index)
        decision = dict(old or dict(source=[], model=[result['tokens'][index]],
                                    chosen=[result['tokens'][index]], status='UNRESOLVED'))
        choice = choose(result['tokens'][index], decision)
        decision['publication'] = choice
        if choice['tier'] == 'flagged':
            flags[index] = choice
            result['tokens'][index] = choice['reading']
            decision['source_image_only_before_publication'] = decision.get('source_image_only', False)
            decision['source_image_only'] = False
        result.setdefault('word_notes', {})[index] = decision
    result['flagged_readings'] = flags
    return flags


def flagged_html(reading, image, target, style=''):
    """Print evidence is decorative to speech; the live reading is spoken once."""
    metric = ' style="' + escape(style, quote=True) + '"' if style else ''
    return ('<span class="reflow-flagged-reading"><span class="reflow-plausible">'
            + escape(reading) + '</span> '
            + '<a class="source-word-image" href="#' + target + '"><img src="'
            + image + '" alt=""' + metric + '/></a></span>')


def stylesheet():
    return '''\n.reflow-plausible { border-bottom:1px solid currentColor; }
.reflow-flagged-reading { font-style:inherit; }
a.source-word-image, a.source-disclosure-ref { text-decoration:none; color:inherit; }
a.source-word-image img { max-width:none; width:auto; }
a.source-disclosure-ref sup { font-size:.55em; line-height:0; }
'''


def join_letter_spaced_readings(result, witness):
    """One whole pixel word may own a marked run of typographic letter atoms.

    The letters themselves are conserved. A complete high-confidence reading at
    this precise row supplies the word boundary; spelling elsewhere does not.
    Existing opaque-code refusals survive as a single print-only word.
    """
    from .pixel_adjudication import _complete_word_occurrence
    tokens=result['tokens']; groups=[];i=0
    while i<len(tokens):
        candidates=[]
        for end in range(i+2,min(i+16,len(tokens))+1):
            indices=list(range(i,end))
            if not all(len(tokens[j])==1 and tokens[j].isalpha()
                       and j in result['marked'] and j in result['boxes'] for j in indices):break
            boxes=[result.get('word_notes',{}).get(j,{}).get('source_bbox',result['boxes'][j])for j in indices]
            if any(not(-.5<=b[0]-a[2]<=.8*max(a[3]-a[1],b[3]-b[1])
                       and min(a[3],b[3])-max(a[1],b[1])>=.5*min(a[3]-a[1],b[3]-b[1]))
                   for a,b in zip(boxes,boxes[1:])):continue
            literal=''.join(tokens[i:end]);bounds=[min(b[0]for b in boxes),min(b[1]for b in boxes),
                max(b[2]for b in boxes),max(b[3]for b in boxes)]
            proof=[w for w in witness if equivalent(w['text'].casefold(),literal.casefold())
                   and w.get('confidence',0)>=85 and _complete_word_occurrence(dict(bbox=bounds),w)]
            if len(proof)==1:candidates.append((end,literal,bounds))
        if candidates:
            end,literal,bounds=max(candidates)
            old=[result.get('word_notes',{}).get(j,{})for j in range(i,end)]
            opaque=any(d.get('method')=='unproved_native_character_map' for d in old)
            note=dict(source=[s for d in old for s in d.get('source',[])],model=[literal],chosen=[literal],
                status='UNRESOLVED',method='unproved_native_character_map' if opaque else 'letter_spaced_printed_word',
                source_bbox=bounds,source_image_only=opaque,letter_spaced_atoms=list(range(i,end)))
            groups.append((i,end,literal,bounds,note));i=end
        else:i+=1
    if not groups:return
    starts={a:(z,text,box,note)for a,z,text,box,note in groups}
    output=[];boxes={};notes={};marked=set();verified=[];mapping={};i=0
    while i<len(tokens):
        new=len(output)
        if i in starts:
            end,text,box,note=starts[i];output.append(text);boxes[new]=box;notes[new]=note;marked.add(new)
            mapping.update({old:new for old in range(i,end)});i=end;continue
        output.append(tokens[i]);mapping[i]=new
        if i in result['boxes']:boxes[new]=result['boxes'][i]
        if i in result.get('word_notes',{}):notes[new]=result['word_notes'][i]
        if i in result['marked']:marked.add(new)
        if i in result.get('verified_unchanged',[]):verified.append(new)
        i+=1
    result.update(tokens=output,boxes=boxes,word_notes=notes,marked=marked,verified_unchanged=verified)
    result.setdefault('guard',[]).append(dict(kind='letter_spaced_printed_words',groups=[dict(
        original_indices=list(range(a,z)),reading=text,source_bbox=box)for a,z,text,box,note in groups]))


def conserve_punctuation_boundaries(result, source, model):
    """A source delimiter moved across whitespace is still one occurrence.

    A local correction may keep a source's glued leading comma while the first
    reader already placed that same comma on the preceding token. Account for
    both source tokens and both model tokens before consuming the duplicate.
    Two genuinely printed delimiters and decimal points are unchanged.
    """
    for index, note in list(result.get('word_notes',{}).items()):
        chosen=result['tokens'][index]
        if (index==0 or not chosen or chosen[0] not in ',.;:'
                or not chosen[1:].isalpha() or note.get('status')!='RESOLVED'):
            continue
        a,z=note.get('source_range',[-1,-1]);b,y=note.get('model_range',[-1,-1])
        if not(0<a<z<=len(source) and z-a==1 and 0<b<y<=len(model) and y-b==1):continue
        glyph=chosen[0];left,right=source[a-1],source[a]
        if (left['text'].endswith(tuple(',.;:')) or right['text'][:1]not in ',.;:'
                or model[b]!=chosen[1:] or not model[b-1].endswith(glyph)
                or not result['tokens'][index-1].endswith(glyph)):
            continue
        u,v=left['bbox'],right['bbox']
        if not(min(u[3],v[3])>max(u[1],v[1]) and -.5<=v[0]-u[2]<=2*max(u[3]-u[1],v[3]-v[1])):continue
        result['tokens'][index]=chosen[1:]
        result['marked'].discard(index);result['word_notes'].pop(index,None)
        result.setdefault('guard',[]).append(dict(kind='source_punctuation_boundary_conserved',
            index=index,original_chosen=chosen,reading=chosen[1:],source_range=[a-1,z],model_range=[b-1,y]))
