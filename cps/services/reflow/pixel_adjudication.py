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


def _accent_conflict(left,right):
    """An English pixel read cannot prove a different diacritic on equal letters."""
    def letters(value):
        return ''.join(c for c in unicodedata.normalize('NFD',value.casefold())
                       if c.isalnum() or unicodedata.combining(c))
    a,b=letters(left),letters(right)
    bare=lambda value:''.join(c for c in value if not unicodedata.combining(c))
    return a!=b and bare(a)==bare(b)


def _boundary_parts(model,source):
    """Insert proved spaces while preserving every original model codepoint."""
    parts=[];start=cursor=0
    for word in source[:-1]:
        count=sum(c.isalnum() for c in unicodedata.normalize('NFKD',word))
        seen=0
        while cursor<len(model) and seen<count:
            seen+=sum(c.isalnum() for c in unicodedata.normalize('NFKD',model[cursor]))
            cursor+=1
        while cursor<len(model) and unicodedata.combining(model[cursor]):cursor+=1
        parts.append(model[start:cursor]);start=cursor
    parts.append(model[start:])
    return parts


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


def blank_between(display,left,right,regions):
    """Prove absence only from complete original intervening ink, not OCR gaps."""
    from . import extract
    a=left['bbox'];b=right['bbox']
    height=min(a[3]-a[1],b[3]-b[1])
    if min(a[3],b[3])-max(a[1],b[1])<height*.5 or b[0]<=a[2]:return False
    # Concatenated disjoint crops can exclude the very word under dispute.
    envelope=[a[0],min(a[1],b[1]),b[2],max(a[3],b[3])]
    if not any(r[0]<=envelope[0] and r[1]<=envelope[1] and r[2]>=envelope[2] and r[3]>=envelope[3] for r in regions):return False
    box=extract.pymupdf.Rect(a[2]+.1,envelope[1],b[0]-.1,envelope[3])
    if box.width<=0:return False
    pix=display.pixmap(box,scale=4,max_pixels=200_000)
    from PIL import Image
    import io
    gray=Image.open(io.BytesIO(pix.tobytes('png'))).convert('L')
    return gray.getextrema()[0]>=180


def _word_note(record, index, span_id):
    """Keep full edit evidence once; never invent a positional word pairing."""
    if max(len(record[key]) for key in ('source','model','chosen'))<=1:return record
    if (record['chosen']==record['source'] and not record['model'] or
            not record['source'] and record['chosen']==record['model']):
        return dict(record,source=record['source'][index:index+1],
                    model=record['model'][index:index+1],chosen=record['chosen'][index:index+1])
    return dict(source=[],model=[],chosen=record['chosen'][index:index+1],
        status=record['status'],method=record['method'],mapping='unplaced_within_span',
        span_id=span_id,source_range=record.get('source_range'),model_range=record.get('model_range'))


def _terminal_word_view(model, words):
    """Keep split terminal punctuation in its existing lexical token.

    This view changes neither the literal punctuation nor its original input
    records. A distant punctuation instance cannot enlarge the word's source
    box or establish physical continuation proof. The original word inventory
    remains authoritative for that separate geometry decision.
    """
    literals={_key(token) for token in model}
    result=[]
    for word in words:
        suffix=word['text']
        if (result and suffix and all(c in '.,;:!?' for c in suffix)
                and _key(result[-1]['text']+suffix) in literals):
            previous=result[-1]
            a,b=previous['bbox'],word['bbox']
            height=max(a[3]-a[1],b[3]-b[1],1)
            adjacent=(min(a[3],b[3])-max(a[1],b[1])>=height*.5
                      and 0<=b[0]-a[2]<=height*1.5)
            box=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])] if adjacent else a
            result[-1]=dict(previous,text=previous['text']+suffix,bbox=box,
                confidence=min(previous.get('confidence',100),word.get('confidence',100)))
        else:result.append(word)
    return result


def adjudicate(model, source, witness, uncertain=(), absence_proof=None, corroboration=(), unresolved_source=False):
    """Return word-level decisions; ``witness`` is independent crop OCR.

    Empty alternatives require both adjacent anchors, so a missing edge cannot
    prove a deletion. Repeated phrases require a unique surrounding sequence.
    Source rectangles belong to occurrences, rather than spelling lookups.
    """
    source=_terminal_word_view(model,source)
    witness=_terminal_word_view(model,witness)
    corroboration=_terminal_word_view(model,corroboration)
    source_tokens=[w['text'] for w in source]
    visible=[(i,t) for i,t in enumerate(model) if not t.startswith('<n:')]
    proposed=[t for _,t in visible]
    original=[_key(t) for t in source_tokens]
    candidate=[_key(t) for t in proposed]
    pixels=[_key(w['text']) for w in witness]
    changes=[];chosen=[];decisions=[];pending=set(uncertain)
    matcher=SequenceMatcher(None, original, candidate, autojunk=False)
    placement=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())
                               if c.isalnum())
    operations=[]
    for kind,a,z,b,y in matcher.get_opcodes():
        if kind!='replace' or max(z-a,y-b)>32:
            operations.append((kind,a,z,b,y));continue
        folded=SequenceMatcher(None,[placement(t) for t in original[a:z]],
            [placement(t) for t in candidate[b:y]],autojunk=False)
        for tag,i,k,j,l in folded.get_opcodes():
            # Similarity proposes correspondence, never the printed word.
            # The independent physical occurrence must still prove a choice.
            paired=(k-i==l-j and (tag=='equal' or all(
                SequenceMatcher(None,placement(original[a+n]),placement(candidate[b+m])).ratio()>=.65
                for n,m in zip(range(i,k),range(j,l)))))
            if paired:
                operations.extend(('replace',a+n,a+n+1,b+m,b+m+1)
                                  for n,m in zip(range(i,k),range(j,l)))
            else:operations.append(('replace',a+i,a+k,b+j,b+l))
    for kind,a,z,b,y in operations:
        if kind=='equal':
            for token,index in zip(proposed[b:y],range(a,z)):
                own=[word for word in witness if _same_occurrence(source[index],word)]
                word=own[0] if len(own)==1 else None
                second=([other for other in corroboration if _same_occurrence(word,other)
                    and _key(word['text'])==_key(other['text'])] if word else [])
                corrected=bool(word and word.get('confidence',0)>=90 and len(second)==1
                    and _key(word['text'])!=_key(token)
                    and _key(word['text']).rstrip('.,;:!?')!=_key(token).rstrip('.,;:!?')
                    and not _accent_conflict(word['text'],token))
                if corrected:
                    record=dict(source=[token],model=[token],chosen=[word['text']],status='RESOLVED',
                        method='two_own_pixel_readings',source_range=[index,index+1])
                    decisions.append(record);chosen.append((word['text'],word['bbox'],record))
                else:chosen.append((token,source[index]['bbox'],None))
            continue
        if z-a==y-b==1:
            source_word,model_word=original[a],candidate[b]
            stem=source_word.rstrip('.,;:!?')
            if stem and stem==model_word.rstrip('.,;:!?') and stem in (source_word,model_word):
                # Missing terminal punctuation is not a changed lexical word.
                # Preserve the first reading without manufacturing endpoint
                # proof from a source token that does not print that suffix.
                if unresolved_source and source_word!=model_word:
                    record=dict(source=[source_tokens[a]],model=[proposed[b]],chosen=[source_tokens[a]],
                        status='UNRESOLVED',method='retained_source_terminal_punctuation',
                        source_range=[a,z],model_range=[b,y])
                    decisions.append(record);chosen.append((source_tokens[a],source[a]['bbox'],record))
                else:chosen.append((proposed[b],source[a]['bbox'],None))
                continue
        alternatives=[('source',original[a:z],source_tokens[a:z]),
                      ('model',candidate[b:y],proposed[b:y])]
        # At least one nonempty adjacent anchor must locate the edit. Empty
        # alternatives need both; they cannot be proved at a cropped edge.
        left=original[max(0,a-2):a];right=original[z:z+2]
        supported=[]
        direct=[];boundary=None
        if z-a==y-b==1 and not any(len(_matches(pixels,left+seq+right))>1
                                   for name,seq,literal in alternatives):
            own=[w for w in witness if _same_occurrence(source[a],w)]
            if len(own)==1 and own[0].get('confidence',0)>=90:
                word=own[0];key=_key(word['text'])
                second=[w for w in corroboration if _same_occurrence(word,w) and _key(w['text'])==key]
                nearby=source[max(0,a-2):a]+source[z:z+2]
                anchored=any(len([w for w in witness if _same_occurrence(n,w)
                    and _key(n['text'])==_key(w['text']) and w.get('confidence',0)>=85])==1 for n in nearby)
                alternative=key in (original[a],candidate[b])
                similar=max(SequenceMatcher(None,placement(key),placement(t)).ratio()
                            for t in (original[a],candidate[b]))>=.65
                if (anchored or len(second)==1) and (alternative or len(second)==1 and similar):
                    direct=[('pixels',[word['text']],[word])]
        if 1<z-a<=3 and y-b==1 and ''.join(placement(t) for t in original[a:z])==placement(candidate[b]):
            owned=[[w for w in witness if _same_occurrence(s,w)] for s in source[a:z]]
            if all(len(row)==1 for row in owned):
                evidence=[row[0] for row in owned]
                distinct=len({id(w) for w in evidence})==len(evidence)
                literal=all(placement(w['text'])==placement(s['text'])
                            and w.get('confidence',0)>=90 for s,w in zip(source[a:z],evidence))
                neighbors=source[max(0,a-2):a]+source[z:z+2]
                anchored=any(len([w for w in witness if _same_occurrence(n,w)
                    and _key(n['text'])==_key(w['text']) and w.get('confidence',0)>=85])==1 for n in neighbors)
                if distinct and literal and anchored:
                    parts=_boundary_parts(proposed[b],source_tokens[a:z])
                    boundary=('boundary',parts,evidence)
                    if parts==source_tokens[a:z] and all(_key(s['text'])==_key(w['text']) for s,w in zip(source[a:z],evidence)):
                        direct=[('source',parts,evidence)]
        for name,seq,literal in ([] if direct else alternatives):
            if not (left or right) or not seq and not (left and right):continue
            locations=_matches(pixels,left+seq+right)
            if len(locations)!=1:continue
            context=witness[locations[0]:locations[0]+len(left+seq+right)]
            anchors=list(zip(source[max(0,a-2):a],context[:len(left)]))
            if right:anchors+=list(zip(source[z:z+2],context[-len(right):]))
            if not anchors or not all(_same_occurrence(a,b) for a,b in anchors):continue
            if not seq and (absence_proof is None or not absence_proof(source[a-1],source[z])):continue
            start=locations[0]+len(left)
            evidence=witness[start:start+len(seq)]
            if any(w.get('confidence',0)<85 for w in witness[locations[0]:locations[0]+len(left+seq+right)]):continue
            supported.append((name,literal,evidence))
        supported=direct or supported
        # A changed accent is not certified by an English recognizer's missing
        # or different mark. Preserve the first reader's codepoints and disclose
        # the unresolved choice, even when the bare letter sequence agrees.
        supported=[choice for choice in supported if not _accent_conflict(
            ''.join(choice[1]),''.join(proposed[b:y]))]
        # A row transition alone does not print a division. Preserve each
        # independently witnessed word boundary unless an actual terminal
        # hyphen in the original source proves the joined spelling.
        wrapped=(z-a>1 and y-b==1 and
                 any(t.endswith(('-', '\u00ad')) for t in source_tokens[a:z-1])
                 and ''.join(placement(t) for t in source_tokens[a:z])==placement(proposed[b]))
        # A crop that ends at a divided atom has not observed the complete
        # lexical word. Matching that prefix cannot certify deleting the
        # continuation from a complete first reading.
        partial_wrap=(z-a==y-b==1 and source_tokens[a].endswith(('-', '\u00ad'))
            and placement(proposed[b]).startswith(placement(source_tokens[a]))
            and len(placement(proposed[b]))>len(placement(source_tokens[a])))
        if wrapped or partial_wrap:supported=[];boundary=None
        resolved=len(supported)==1
        name,literal,evidence=(supported[0] if resolved else
            boundary if boundary else ('source',source_tokens[a:z],[]) if unresolved_source or not proposed[b:y] else ('model',proposed[b:y],[]))
        record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=list(literal),
                    status='RESOLVED' if resolved else 'UNRESOLVED',
                    method='independent_upright_crop' if resolved else 'pixel_word_boundary_only' if boundary else 'pixel_read_inconclusive',
                    source_range=[a,z],model_range=[b,y])
        decisions.append(record)
        boxes=([w['bbox'] for w in evidence] if evidence else
               [w['bbox'] for w in source[a:z]])
        box=( [min(x[0] for x in boxes),min(x[1] for x in boxes),
               max(x[2] for x in boxes),max(x[3] for x in boxes)] if boxes else None)
        for j,token in enumerate(literal):
            word_record=_word_note(record,j,len(decisions)-1)
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


class CropWitness(list):
    """An unavailable second recognition does not invalidate other evidence."""
    failure = None


def read_crop(display, regions, cache):
    import subprocess
    try:
        return _read_crop(display, regions, cache)
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result = CropWitness()
        result.failure = f'{type(error).__name__}: {error}'
        return result


def _read_crop(display, regions, cache):
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
    if not os.path.isfile(engine) or not os.access(engine,os.X_OK):
        raise RuntimeError('Independent OCR executable is unavailable')
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


def refine_fallback(tokens, witness, lexicon, source=(), corroboration=()):
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
    folded=SequenceMatcher(None,[t.casefold() for t in keys],
                           [t.casefold() for t in pixels],autojunk=False)
    if sum(n for _,_,n in folded.get_matching_blocks())/max(1,len(keys))<.75:return None
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
        owned_span=(len(tokens[a:z])==len(evidence) and bool(evidence) and all(
            i in source_map and _same_occurrence(source_map[i],word)
            for i,word in zip(range(a,z),evidence)))
        # Case changes preserve the lexical word. An unfamiliar spelling needs
        # two pixel readings at this occurrence, never a dictionary preference
        # or another copy elsewhere. Local OCR remains explicitly unverified.
        case_only=owned_span and all(_key(t).casefold()==_key(w['text']).casefold()
                                     for t,w in zip(tokens[a:z],evidence))
        two_views=owned_span and all(len([other for other in corroboration
            if _same_occurrence(word,other) and _key(other['text'])==_key(word['text'])])==1
            for word in evidence)
        resolved=bool(positive and ((unique and anchored and (lexical or numeral))
                                   or case_only or two_views))
        if _accent_conflict(''.join(tokens[a:z]),''.join(w['text'] for w in evidence)):resolved=False
        chosen=[w['text'] for w in evidence] if resolved else tokens[a:z]
        record=dict(source=tokens[a:z],model=tokens[a:z],chosen=chosen,
            status='RESOLVED' if resolved else 'UNRESOLVED',
            method=('two_local_pixel_readings' if two_views else 'pixel_case_agreement' if case_only
                    else 'contextual_crop_lexical_support') if resolved else 'pixel_read_inconclusive')
        decisions.append(record)
        for i,t in enumerate(chosen):
            index=len(output);output.append(t);notes[index]=_word_note(record,i,len(decisions)-1)
            if resolved:bounds[index]=evidence[i]['bbox']
    return dict(tokens=output,boxes=bounds,marked=set(notes),word_notes=notes,decisions=decisions)


def audit_agreement(tokens, source, witness, boundary_proof=None):
    """Flag pixel conflicts even where a layer and the first reading agree.

    A recognizer's different spelling is evidence of a conflict, not authority
    to replace a valid source word. Only the source occurrence's own overlapping
    ink can cause a mark; a different copy of a repeated word cannot do so.
    """
    visible=[(i,token)for i,token in enumerate(tokens) if not token.startswith('<n:')]
    if len(visible)!=len(source):return {}
    # Recognition may fuse adjacent printed words. Exact letters plus an
    # independently observed blank between every original occurrence can prove
    # the existing spaces. Geometry alone and a dictionary never waive a clash.
    segmented=set()
    if boundary_proof is not None:
        for other in witness:
            owned=[(j,w)for j,w in enumerate(source)if _same_occurrence(w,other)]
            if not 2<=len(owned)<=3 or other.get('confidence',0)<85:continue
            if ''.join(_key(w['text'])for j,w in owned)!=_key(other['text']):continue
            if all(boundary_proof(a,b)for (_,a),(_,b)in zip(owned,owned[1:])):
                segmented.update(j for j,w in owned)
    notes={}
    for j,((i,token),word) in enumerate(zip(visible,source)):
        if j in segmented:continue
        overlapping=[w for w in witness if _same_occurrence(word,w)]
        if len(overlapping)!=1:continue
        other=overlapping[0]
        if other.get('confidence',0)<85 or _key(other['text'])==_key(token):continue
        # Missing terminal punctuation is not a contradictory word reading.
        # Keep the original punctuation; letter, case and internal-symbol
        # differences still require occurrence-linked uncertainty.
        if _key(other['text']).rstrip('.,;:!?')==_key(token).rstrip('.,;:!?'):continue
        notes[i]=dict(source=[token],model=[token],chosen=[token],
            alternative=[other['text']],status='UNRESOLVED',method='agreeing_readings_pixel_conflict')
    return notes


def fragment_word_ranges(fragment, tokens):
    """Bind the complete authored stream across inline tags to raw ranges.

    A formatted word may enclose inline markup in its range. Callers retain that
    markup when adding a control and must not replace it as a plain text slice.
    Different inventories are refused; equal spelling never selects an owner.
    """
    import re
    from . import annotate
    plain=[];starts=[];ends=[];previous=None
    for node in annotate._text_nodes(fragment,include_uncertain=True):
        if previous is not None and re.search(r'</li\s*>',fragment[previous:node.raw_start]):
            plain.append(' ');starts.append(previous);ends.append(previous)
        plain.extend(node.plain);starts.extend(node.starts);ends.extend(node.ends)
        previous=node.ends[-1] if node.ends else previous
    words=[]
    for word in re.finditer(r'\S+',''.join(plain)):
        a,z=starts[word.start()],ends[word.end()-1]
        opened=[]
        for closing,name in re.findall(r'<(/?)(sup|sub|em|strong|i|b|a|span)\b[^>]*>',fragment[a:z],re.I):
            if closing:
                if opened and opened[-1]==name.casefold():opened.pop()
            else:opened.append(name.casefold())
        while opened:
            closing=re.match(r'</'+opened[-1]+r'\s*>',fragment[z:],re.I)
            if closing is None:break
            z+=closing.end();opened.pop()
        words.append((word.group(),a,z))
    if [word for word,_,_ in words]!=tokens:return None
    return [(a,z) for _,a,z in words]
