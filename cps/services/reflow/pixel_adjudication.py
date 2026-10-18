"""Settle individual transcription edits using an independent pixel witness.

A native layer and a model can disagree without either one being authoritative.
Only a uniquely context-bound reading from the second recognition can resolve an
edit. Unresolved edits retain their source-linked uncertainty; dictionary words
alone never certify a substitution, omission or insertion.
"""
from difflib import SequenceMatcher
import unicodedata


def _key(value):
    return unicodedata.normalize('NFKC', value).replace('\u00ad', '').translate(
        str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))


def _matches(words, sequence):
    if not sequence:return []
    return [i for i in range(len(words)-len(sequence)+1)
            if words[i:i+len(sequence)] == sequence]


def _same_occurrence(left,right):
    a=left['bbox'];b=right['bbox']
    height=max(a[3]-a[1],b[3]-b[1],1)
    return (abs((a[1]+a[3]-b[1]-b[3])/2)<=height*.75+2
            and min(a[3],b[3])>max(a[1],b[1])
            and min(a[2],b[2])-max(a[0],b[0])>=.3*min(a[2]-a[0],b[2]-b[0]))


def adjudicate(model, source, witness, uncertain=()):
    """Return word-level decisions; ``witness`` is independent crop OCR.

    Empty alternatives require both adjacent anchors, so a missing edge cannot
    prove a deletion. Repeated phrases require a unique surrounding sequence.
    Source rectangles belong to occurrences, rather than spelling lookups.
    """
    source_tokens=[w['text'] for w in source]
    visible=[(i,t) for i,t in enumerate(model) if not t.startswith('<n:')]
    proposed=[t for _,t in visible]
    original=[_key(t) for t in source_tokens]
    candidate=[_key(t) for t in proposed]
    pixels=[_key(w['text']) for w in witness]
    changes=[];chosen=[];decisions=[];pending=set(uncertain)
    matcher=SequenceMatcher(None, original, candidate, autojunk=False)
    for kind,a,z,b,y in matcher.get_opcodes():
        if kind=='equal':
            chosen.extend((t, source[j]['bbox'], None) for t,j in zip(proposed[b:y],range(a,z)))
            continue
        alternatives=[('source',original[a:z],source_tokens[a:z]),
                      ('model',candidate[b:y],proposed[b:y])]
        # At least one nonempty adjacent anchor must locate the edit. Empty
        # alternatives need both; they cannot be proved at a cropped edge.
        left=original[max(0,a-2):a];right=original[z:z+2]
        supported=[]
        for name,seq,literal in alternatives:
            if not (left or right) or not seq and not (left and right):continue
            locations=_matches(pixels,left+seq+right)
            if len(locations)!=1:continue
            context=witness[locations[0]:locations[0]+len(left+seq+right)]
            anchors=list(zip(source[max(0,a-2):a],context[:len(left)]))
            if right:anchors+=list(zip(source[z:z+2],context[-len(right):]))
            if not anchors or not all(_same_occurrence(a,b) for a,b in anchors):continue
            start=locations[0]+len(left)
            evidence=witness[start:start+len(seq)]
            if any(w.get('confidence',0)<85 for w in witness[locations[0]:locations[0]+len(left+seq+right)]):continue
            supported.append((name,literal,evidence))
        resolved=len(supported)==1
        name,literal,evidence=(supported[0] if resolved else
            ('model',proposed[b:y],[]))
        record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=list(literal),
                    status='RESOLVED' if resolved else 'UNRESOLVED',
                    method='independent_upright_crop' if resolved else 'pixel_read_inconclusive',
                    source_range=[a,z],model_range=[b,y])
        decisions.append(record)
        boxes=([w['bbox'] for w in evidence] if evidence else
               [w['bbox'] for w in source[a:z]])
        box=( [min(x[0] for x in boxes),min(x[1] for x in boxes),
               max(x[2] for x in boxes),max(x[3] for x in boxes)] if boxes else None)
        for j,token in enumerate(literal):
            word_record=(dict(record,source=record['source'][j:j+1],
                model=record['model'][j:j+1],chosen=[token])
                if len(record['source'])==len(record['chosen']) and len(record['chosen'])>1 else record)
            chosen.append((token,evidence[j]['bbox'] if evidence else box,word_record))
        # A deletion still needs a control attached to an adjacent live word.
        if not literal:changes.append((len(chosen),record,box))
    # Reinsert source-proven note placeholders at their original model boundary.
    # Lexical edits never manufacture, delete or renumber a note reference.
    tokens=[t for t,_,_ in chosen];bounds={i:box for i,(_,box,_) in enumerate(chosen) if box}
    marks={i for i,(_,_,record) in enumerate(chosen) if record}
    notes={i:record for i,(_,_,record) in enumerate(chosen) if record}
    for at,record,box in changes:
        if not tokens:continue
        index=min(at,len(tokens)-1);marks.add(index);notes[index]=record
        if box:bounds[index]=box
    # Uncertainty positions are translated through the model/chosen alignment.
    translation=SequenceMatcher(None,proposed,tokens,autojunk=False)
    verified=[]
    for kind,a,z,b,y in translation.get_opcodes():
        affected=([i for j,i in zip(range(a,z),range(b,y)) if visible[j][0] in pending]
            if kind=='equal' else list(range(b,y)) if any(visible[j][0] in pending for j in range(a,z)) else [])
        for i in affected:
            # A reader's low-confidence flag is not permanent uncertainty when
            # the same occurrence has an exact, uniquely contextual pixel read.
            left=max(0,i-1);right=min(len(tokens),i+2)
            locations=_matches(pixels,[_key(t) for t in tokens[left:right]])
            context=witness[locations[0]:locations[0]+right-left] if len(locations)==1 else []
            if i not in notes and context and all(w.get('confidence',0)>=90 for w in context) and all(
                    j in bounds and _same_occurrence(dict(bbox=bounds[j]),word)
                    for j,word in zip(range(left,right),context)):
                verified.append(i);continue
            marks.add(i)
            notes.setdefault(i,dict(source=[tokens[i]],model=[tokens[i]],chosen=[tokens[i]],
                status='UNRESOLVED',method='reader_uncertainty'))
    consumed=0;inserted=0
    for i,token in enumerate(model):
        if token.startswith('<n:'):
            at=0
            for _,a,z,b,y in translation.get_opcodes():
                if a<=consumed<=z:
                    at=b+min(y-b,max(0,consumed-a));break
            at+=inserted;inserted+=1
            tokens.insert(at,token)
            bounds={j+(j>=at):v for j,v in bounds.items()}
            marks={j+(j>=at) for j in marks};notes={j+(j>=at):v for j,v in notes.items()}
        else:consumed+=1
    return dict(tokens=tokens,boxes=bounds,marked=marks,decisions=decisions,word_notes=notes,verified_unchanged=verified)


def _engine_identity(engine):
    """Bind cached witnesses to the recognizer and installed language data."""
    return _cached_engine_identity(engine)


from functools import lru_cache
@lru_cache(maxsize=4)
def _cached_engine_identity(engine):
    import hashlib
    import os
    import re
    import subprocess
    from pathlib import Path
    version=subprocess.run([engine,'--version'],capture_output=True,timeout=5,check=True)
    languages=subprocess.run([engine,'--list-langs'],capture_output=True,timeout=5,check=True)
    digest=hashlib.sha256(version.stdout+version.stderr)
    listing=(languages.stdout+languages.stderr).decode('utf-8',errors='replace')
    roots=[os.environ.get('TESSDATA_PREFIX','')]
    roots.extend(re.findall(r'"([^"\n]+)"',listing))
    found=False
    for root in roots:
        if not root:continue
        for candidate in (Path(root)/'eng.traineddata',Path(root)/'tessdata/eng.traineddata'):
            if candidate.is_file():
                digest.update(candidate.read_bytes());found=True;break
        if found:break
    if not found:raise RuntimeError('Cannot fingerprint independent OCR language data')
    return digest.digest()


def read_crop(display, regions, cache):
    """Recognize upright source crops with bounded pixels and a separate layout.

    The full-page recognizer uses 3x/automatic page layout. This witness uses
    4x/single-block layout, cached by encoded pixels and recognition settings.
    A multi-region block is read in source-region order rather than flattening
    a bounding rectangle across unrelated columns.
    """
    import csv
    import fcntl
    import hashlib
    import io
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    engine=shutil.which('tesseract')
    if not engine:raise RuntimeError('Pixel adjudication requires local OCR')
    root=Path(cache);root.mkdir(parents=True,exist_ok=True);output=[]
    for region in regions:
        from . import extract
        box=extract.pymupdf.Rect(region)
        box.x0-=2;box.y0-=2;box.x1+=2;box.y1+=2;box &= display.rect
        pix=display.pixmap(box,scale=4,max_pixels=4_000_000)
        image=pix.tobytes('png');identity=hashlib.sha256(b'upright-block-eng-psm6-v2\0'+_engine_identity(engine)+image).hexdigest()
        target=root/(identity+'.json')
        with (root/(identity+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists():rows=json.loads(target.read_text())
            else:
                result=subprocess.run([engine,'stdin','stdout','-l','eng','--psm','6','tsv'],
                    input=image,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
                if result.returncode:raise RuntimeError('Independent crop recognition failed')
                rows=[]
                for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')),
                    delimiter='\t',quoting=csv.QUOTE_NONE):
                    if row['level']!='5' or not row['text'].strip():continue
                    x,y,w,h=[int(row[k]) for k in ('left','top','width','height')]
                    rows.append(dict(text=row['text'],confidence=float(row['conf']),
                        bbox=[x/pix.width,y/pix.height,(x+w)/pix.width,(y+h)/pix.height]))
                temp=target.with_suffix('.'+str(os.getpid())+'.tmp')
                temp.write_text(json.dumps(rows,ensure_ascii=False));os.replace(temp,target)
        for row in rows:
            a,b,c,d=row['bbox']
            output.append(dict(row,bbox=[box.x0+a*box.width,box.y0+b*box.height,
                                         box.x0+c*box.width,box.y0+d*box.height]))
    return output


def refine_fallback(tokens, witness, lexicon, source=()):
    """Resolve a damaged-layer fallback with positive, contextual pixel ink.

    Lexical support is secondary to a unique crop occurrence. It can support an
    OCR misspelling, but cannot rewrite an unfamiliar name into a common word or
    delete a real word simply because the dictionary prefers another sentence.
    """
    import re
    if any(t.startswith('<n:') for t in tokens) or not lexicon or not source:return None
    keys=[_key(t) for t in tokens];pixels=[_key(w['text']) for w in witness]
    matcher=SequenceMatcher(None,keys,pixels,autojunk=False)
    source_map={j:source[i] for a,b,n in SequenceMatcher(None,[_key(w['text']) for w in source],keys,autojunk=False).get_matching_blocks() for i,j in zip(range(a,a+n),range(b,b+n))}
    if sum(n for _,_,n in matcher.get_matching_blocks())/max(1,len(keys))<.75:return None
    known=lambda token:re.sub(r"[^\w'’]",'',token).casefold() in lexicon
    output=[];notes={};bounds={};decisions=[]
    for kind,a,z,b,y in matcher.get_opcodes():
        if kind=='equal':
            for i,j in zip(range(a,z),range(b,y)):
                if i in source_map:bounds[len(output)]=source_map[i]['bbox']
                output.append(tokens[i])
            continue
        left=keys[max(0,a-2):a];right=keys[z:z+2]
        evidence=witness[b:y]
        positive=bool(evidence) and all(w.get('confidence',0)>=90 for w in evidence)
        unique=len(_matches(pixels,left+pixels[b:y]+right))==1
        anchors=list(zip(range(max(0,a-2),a),witness[max(0,b-len(left)):b]))+list(zip(range(z,z+len(right)),witness[y:y+len(right)]))
        anchored=bool(left and right) and all(i in source_map and _same_occurrence(source_map[i],word) for i,word in anchors)
        lexical=(all(known(w['text']) for w in evidence) and
                 (not tokens[a:z] or
                  all(not any(c.isupper() for c in t) for t in tokens[a:z]) and
                  any(not known(t) for t in tokens[a:z])))
        numeral=(len(tokens[a:z])==len(evidence)==1 and
                 re.fullmatch(r'[IVXLCDM]+|[0-9]+',tokens[a]) and
                 re.fullmatch(r'[IVXLCDM]+|[0-9]+',evidence[0]['text']))
        resolved=bool(positive and unique and anchored and (lexical or numeral))
        chosen=[w['text'] for w in evidence] if resolved else tokens[a:z]
        record=dict(source=tokens[a:z],model=tokens[a:z],chosen=chosen,
            status='RESOLVED' if resolved else 'UNRESOLVED',
            method='contextual_crop_lexical_support' if resolved else 'pixel_read_inconclusive')
        decisions.append(record)
        for i,t in enumerate(chosen):
            index=len(output);output.append(t);notes[index]=record
            if resolved:bounds[index]=evidence[i]['bbox']
    return dict(tokens=output,boxes=bounds,marked=set(notes),word_notes=notes,decisions=decisions)


def audit_agreement(tokens, source, witness):
    """Flag pixel conflicts even where a layer and the first reading agree.

    A recognizer's different spelling is evidence of a conflict, not authority
    to replace a valid source word. Only the source occurrence's own overlapping
    ink can cause a mark; a different copy of a repeated word cannot do so.
    """
    if len(tokens)!=len(source):return {}
    notes={}
    for i,(token,word) in enumerate(zip(tokens,source)):
        overlapping=[w for w in witness if _same_occurrence(word,w)]
        if len(overlapping)!=1:continue
        other=overlapping[0]
        if other.get('confidence',0)<85 or _key(other['text'])==_key(token):continue
        notes[i]=dict(source=[token],model=[token],chosen=[token],
            alternative=[other['text']],status='UNRESOLVED',method='agreeing_readings_pixel_conflict')
    return notes


def fragment_word_ranges(fragment, tokens):
    """Bind a complete simple fragment to exact raw word ranges, or refuse.

    Formatting inside a word, generated controls and differing token inventories
    require a richer binding. Refusal is explicit; spelling searches never place
    a warning on the first equal word elsewhere in the fragment.
    """
    import re
    from . import annotate
    words=[]
    for node in annotate._text_nodes(fragment,include_uncertain=True):
        for word in re.finditer(r'\S+',node.plain):
            words.append((word.group(),node.starts[word.start()],node.ends[word.end()-1]))
    if [word for word,_,_ in words]!=tokens:return None
    return [(a,z) for _,a,z in words]
