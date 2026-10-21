"""Settle individual transcription edits using an independent pixel witness.

A native layer and a model can disagree without either one being authoritative.
Complete, occurrence-bound evidence may correct the first reading. Missing or
partial recognition cannot dispute an otherwise confident complete vision word.
Explicit doubt and conflicting complete evidence stay linked to the source; a
dictionary word alone never certifies a substitution, omission or insertion.
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


def _quote_conflict(left,right):
    """Recognizers that omit a small mark cannot prove its absence in print."""
    def signature(value):
        letters=[];marks=[]
        for char in _key(value):
            if char.isalnum():letters.append(char.casefold())
            elif char in ('"', "'") or unicodedata.category(char) in ('Pi','Pf'):
                marks.append((len(letters),char))
        return ''.join(letters),marks
    a,b=signature(left),signature(right)
    return a[0]==b[0] and a[1]!=b[1]


def _initialism_mark_conflict(left,right,chosen=None):
    """Omitted internal dots are not proof that printed initials lack them."""
    import re
    outer='"“”‘’\',;:!?()[]'
    a,b=(_key(value).strip(outer)for value in (left,right))
    initialism=r'(?:[A-Z]\.){2,}[A-Z]?\.?'
    if not any(re.fullmatch(initialism,value)for value in (a,b)):return False
    letters=lambda value:value.replace('.','').casefold()
    if letters(a)!=letters(b):return False
    def dots(value):
        count=0;positions=set()
        for char in value:
            if char=='.':positions.add(count)
            else:count+=1
        return positions
    required=dots(a)|dots(b)
    if chosen is None:return bool(required)
    candidate=_key(chosen).strip(outer)
    return letters(candidate)!=letters(a)or not required<=dots(candidate)


def _ordinal(value):
    import re
    match=re.fullmatch(r'([0-9]+)(st|nd|rd|th)[.,;:!?]?',_key(value))
    if not match:return False
    number=int(match[1]);suffix=('th'if 10<=number%100<=20 else {1:'st',2:'nd',3:'rd'}.get(number%10,'th'))
    return match[2]==suffix


def _ordinal_noise(value):
    import re
    match=re.fullmatch(r'([0-9gGsSiIlLoOuUbBrRaA]+)(?:st|nd|rd|th)[.,;:!?]?',_key(value))
    return bool(match and (len(match[1])==1 or any(c.isdigit()for c in match[1])))


def _own_word_choice(source,witness):
    reader=getattr(witness,'word_readings',None)
    if reader is None:return None
    return _paired_word_choice(source,reader(source))


def _paired_word_choice(source,views):
    """Each pair uses separate language networks on the complete occurrence."""
    if not views or len(views)%2:return None
    choices={}
    for start in range(0,len(views),2):
        own=[_whole_view_word(source,view) for view in views[start:start+2]]
        if not all(len(row)==1 and row[0].get('confidence',0)>=85 for row in own):continue
        first,second=own[0][0],own[1][0]
        if _key(first['text'])==_key(second['text']):choices[_key(first['text'])]=first
    return next(iter(choices.values()))if len(choices)==1 else None


def _whole_view_word(source,view):
    """A confident tail cannot discard other ink inside its owner's bounds."""
    a=source['bbox']
    owned=[w for w in view if a[0]-.5<=(w['bbox'][0]+w['bbox'][2])/2<=a[2]+.5
           and min(a[3],w['bbox'][3])>max(a[1],w['bbox'][1])]
    if len(owned)!=1:return []
    return owned if _complete_recognition(source,owned[0]) else []


def _model_word_choice(source,witness,model):
    """An intact independent read can corroborate, rather than override, a model.

    Split fragments and unavailable networks supply no competing complete word.
    A credible complete alternative keeps the disagreement unresolved. Repeated
    nonword noise from one trained network cannot veto positive complete pixel
    support for the first reader; agreement from both networks still can.
    """
    reader=getattr(witness,'word_readings',None)
    if reader is None:return None
    views=reader(source)
    own=[row[0] for view in views if (row:=_whole_view_word(source,view))
         and row[0].get('confidence',0)>=85]
    if any(_key(w['text'])!=_key(model)and not _dictionary_regression(model,w['text'])
           for w in own):return None
    for start in range(0,len(views),2):
        competing=_paired_word_choice(source,views[start:start+2])
        if competing and _key(competing['text'])!=_key(model):return None
    return next((w for w in own if _key(w['text'])==_key(model)
                 and w.get('confidence',0)>=90),None)


def _optical_counterread(source,witness,model,scope=None):
    """Missing or split recognition is not a competing complete printed word."""
    views=[witness]
    for name in ('word_readings','resampled','alternates'):
        reader=getattr(witness,name,None)
        if reader is not None:views.extend(reader(source)if name=='word_readings'else reader())
    for view in views:
        row=_whole_view_word(source,view)
        if not row:
            # A padded source box can enclose a complete equally long word.
            # This is counterevidence, not authority to overwrite the reader;
            # a shorter tail still cannot challenge a complete lexical word.
            a=source['bbox']
            owned=[w for w in view if a[0]-.5<=(w['bbox'][0]+w['bbox'][2])/2<=a[2]+.5
                   and min(a[3],w['bbox'][3])>max(a[1],w['bbox'][1])]
            if len(owned)==1:
                w=owned[0]
                if (sum(c.isalnum()for c in w['text'])==sum(c.isalnum()for c in model)
                    and _complete_word_occurrence(source,w)
                    and min(a[3],w['bbox'][3])-max(a[1],w['bbox'][1])>=.55*(a[3]-a[1])):
                    row=owned
        if (row and row[0].get('confidence',0)>=85 and _key(row[0]['text'])!=_key(model)
                and (scope is None or _covers_ink_scope(row,scope))):return True
    return False


def _small_mark_conflict(left,right):
    """Inconclusive recognition cannot settle accents, case or added ink marks."""
    def marks(value):
        count=0;result=[]
        for char in _key(value):
            if char.isalnum():count+=1
            else:result.append((count,char))
        return result
    a,b=_key(left),_key(right)
    return (_accent_conflict(a,b) or a!=b and a.casefold()==b.casefold()
            or marks(a)!=marks(b))


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
        closing=bool(suffix and all(c in '"”’'for c in suffix))
        if (result and suffix and all(c in '.,;:!?"”’' for c in suffix)
                and _key(result[-1]['text']+suffix) in literals):
            previous=result[-1]
            a,b=previous['bbox'],word['bbox']
            height=max(a[3]-a[1],b[3]-b[1],1)
            adjacent=(min(a[3],b[3])-max(a[1],b[1])>=height*.5
                      and 0<=b[0]-a[2]<=height*1.5)
            if closing:
                # A closing quotation may sit above the lexical baseline. Its
                # own printed extent and the exact model suffix must agree;
                # a distant mark cannot be absorbed into this occurrence.
                adjacent=(min(a[3],b[3])-max(a[1],b[1])>=.5*min(a[3]-a[1],b[3]-b[1])
                    and 0<=b[0]-a[2]<=height*1.5)
                if not adjacent:
                    result.append(word);continue
            box=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])] if adjacent else a
            result[-1]=dict(previous,text=previous['text']+suffix,bbox=box,
                confidence=min(previous.get('confidence',100),word.get('confidence',100)))
        else:result.append(word)
    if any(hasattr(words,name) for name in ('alternates','resampled','word_readings','word_scope','word_separation','model_reading','failure')):
        result=CropWitness(result)
        for name in ('alternates','resampled','mark_absence','word_readings','word_scope','word_separation','model_reading','failure'):
            if hasattr(words,name):setattr(result,name,getattr(words,name))
    return result


def proved_native_occurrence(token,box,words):
    if box is None:return False
    return len([w for w in words if _key(token)==_key(w['text']) and
                _same_occurrence(dict(bbox=box),w)])==1


def _dictionary_regression(original,chosen):
    """A dictionary vetoes an unsafe nonword substitution; it never selects text."""
    from .text_quality import english_lexicon
    letters=lambda value:''.join(c for c in _key(value).casefold()if c.isalpha())
    a,b=letters(original),letters(chosen)
    if not a or not b or not(a.isascii()and b.isascii()):return False
    vocabulary=english_lexicon()
    return a!=b and a in vocabulary and b not in vocabulary


def _joined_stroke_conflict(original,chosen):
    """Agreement on joined Latin strokes cannot certify a different whole word.

    Both alternatives can be dictionary entries. Keep the original ink until
    evidence beyond the shared recognizer confusion distinguishes the strokes.
    Reliably painted native occurrences take their earlier authority path.
    """
    letters=lambda value:''.join(c for c in _key(value).casefold()if c.isalpha())
    a,b=letters(original),letters(chosen)
    # An alphabetic first reading cannot lose a printed letter to a path slash
    # merely because two alphabet recognizers repeat that partial stroke.
    if _key(original).strip('.,;:!?()[]').isalpha()and any(c in chosen for c in ('/',chr(92))):return True
    if not(a and b and a!=b and a.isascii()and b.isascii()):return False
    joined=(a.replace('rn','m')==b or b.replace('rn','m')==a)
    # Shared alphabet networks can also agree after a crossbar or descender
    # disappears. Complete word boxes do not prove the distinguishing stroke.
    # This veto retains original ink; it never chooses a dictionary spelling.
    eroded=(len(a)==len(b)and all(x==y or (x,y)in(
        ('e','c'),('f','l'),('y','v'),('w','v'))for x,y in zip(a,b)))
    # Both first guesses may already contain the same fused glyph. An
    # unfamiliar changed candidate with a known rn alternative is not settled
    # merely because two alphabet networks repeat m. The lexicon only vetoes
    # certification; it never supplies those missing letters to the output.
    from .text_quality import english_lexicon
    vocabulary=english_lexicon()
    unseen_join=(b not in vocabulary and any(
        b[:i]+'rn'+b[i+1:]in vocabulary for i,c in enumerate(b)if c=='m'))
    return joined or eroded or unseen_join


def _letter_number_identity(value):
    return ''.join(c for c in _key(value)if c.isalnum())


def _symbol_identity_conflict(original,chosen):
    """A mathematical glyph cannot be certified as alphabetic/numeric text."""
    symbols=lambda value:{c for c in value if unicodedata.category(c)in('Sm','So')}
    a,b=symbols(original),symbols(chosen)
    # A symbol and ordinary punctuation both have an empty letter inventory.
    # That equality is not evidence that their visible shapes are the same.
    left,right=_letter_number_identity(original),_letter_number_identity(chosen)
    return a!=b and (not(left or right)or bool(a and b)or left!=right)


def _proved_wrap(source,model,witness,same_reading=False):
    """Conserve exact letters across a positively read printed division."""
    if not 2<=len(source)<=3:return None
    hyphens=('-', '\u00ad', '\u2010', '\u2011')
    if not all(w['text'].endswith(hyphens)for w in source[:-1]):return None
    joined=''.join(w['text'][:-1]for w in source[:-1])+source[-1]['text']
    literal=''.join(w['text']for w in source)
    if _key(model)not in(_key(joined),_key(literal)):return None
    for left,right in zip(source,source[1:]):
        a,b=left['bbox'],right['bbox'];height=max(a[3]-a[1],b[3]-b[1],1)
        next_row=(0<=b[1]-a[3]<=3*height and b[0]<a[0])
        next_column=(b[0]>a[2]+2*height and b[3]<a[1]-2*height)
        if not(next_row or next_column):return None
    # The first pixel reading and strong original occurrences already agree
    # on every letter and on the explicit line division. Joining that exact
    # inventory is formatting, as in the equal-token branch; a weak additional
    # crop does not manufacture a lexical disagreement. Weak originals still
    # require the separate positive recognition below.
    evidence=(list(source)if same_reading and not getattr(witness,'failure',None)
              and all(w.get('confidence',0)>=90 for w in source)else [])
    if not evidence:
        for word in source:
            own=[w for w in witness if _complete_word_occurrence(word,w)]
            if (len(own)!=1 or own[0].get('confidence',0)<90
                    or _key(own[0]['text'])!=_key(word['text'])):
                # A short continuation can be weak in block layout while its
                # own complete occurrence has independent language agreement.
                # Exact letters are still required; this never supplies a
                # guessed continuation or licenses a different printed word.
                complete=_own_word_choice(word,witness)
                if complete is None or _key(complete['text'])!=_key(word['text']):return None
                own=[complete]
            evidence.append(own[0])
    if len({id(w)for w in evidence})!=len(evidence):return None
    boxes=[w['bbox']for w in evidence]
    return [min(b[0]for b in boxes),min(b[1]for b in boxes),
            max(b[2]for b in boxes),max(b[3]for b in boxes)]



def _recognized_letter_groups(source,witness):
    """Regroup exact letter fragments using a complete observed word boundary.

    No spelling comes from this operation. It conserves the source characters
    and does not join ordinary adjacent words or incomplete recognitions.
    """
    groups={};occupied=set()
    for word in witness:
        if word.get('confidence',0)<85:continue
        box=word['bbox']
        indices=[i for i,item in enumerate(source)
            if box[0]-.2<=(item['bbox'][0]+item['bbox'][2])/2<=box[2]+.2
            and box[1]-.2<=(item['bbox'][1]+item['bbox'][3])/2<=box[3]+.2]
        if len(indices)<3 or indices!=list(range(indices[0],indices[-1]+1)):continue
        items=[source[i]for i in indices]
        if any(len(_key(w['text']).strip('.,;:'))>2 for w in items):continue
        literal=''.join(w['text']for w in items)
        if _key(literal).casefold()!=_key(word['text']).casefold():continue
        bounds=[min(w['bbox'][0]for w in items),min(w['bbox'][1]for w in items),
                max(w['bbox'][2]for w in items),max(w['bbox'][3]for w in items)]
        if not _complete_recognition(dict(bbox=bounds),word):continue
        if any(i in occupied for i in indices):return source
        if any(b['bbox'][0]<a['bbox'][2]-.2 or
               abs(b['bbox'][1]-a['bbox'][1])>.5*(bounds[3]-bounds[1])
               for a,b in zip(items,items[1:])):continue
        groups[indices[0]]=(indices[-1],dict(text=literal,bbox=bounds,
            confidence=min(w.get('confidence',0)for w in items)))
        occupied.update(indices)
    result=[]
    for i,word in enumerate(source):
        if i in groups:result.append(groups[i][1])
        elif i not in occupied:result.append(word)
    return result

def adjudicate(model, source, witness, uncertain=(), absence_proof=None, corroboration=(), unresolved_source=False, source_print=False, native_print_words=()):
    """Return word-level decisions; ``witness`` is independent crop OCR.

    Empty alternatives require both adjacent anchors, so a missing edge cannot
    prove a deletion. Repeated phrases require a unique surrounding sequence.
    Source rectangles belong to occurrences, rather than spelling lookups.
    """
    if not source_print and not native_print_words:
        source=_recognized_letter_groups(source,witness)
    source=_terminal_word_view(model,source)
    witness=_terminal_word_view(model,witness)
    corroboration=_terminal_word_view(model,corroboration)
    source_tokens=[w['text'] for w in source]
    visible=[(i,t) for i,t in enumerate(model) if not t.startswith('<n:')]
    proposed=[t for _,t in visible]
    original=[_key(t) for t in source_tokens]
    candidate=[_key(t) for t in proposed]
    pixels=[_key(w['text']) for w in witness]
    changes=[];chosen=[];decisions=[];pending=set(uncertain);formatting=set()
    matcher=SequenceMatcher(None, original, candidate, autojunk=False)
    placement=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())
                               if c.isalnum())
    operations=[]
    for kind,a,z,b,y in matcher.get_opcodes():
        # An unreserved literal reference is still part of its lexical token.
        # Folding away that suffix would align the word while leaving the same
        # glyph's OCR letter as a new insertion. Keep the source fragments in
        # one edit until the dedicated reference verifier can place the glyph.
        unplaced_reference=any(proposed[j].endswith(('*','†','‡'))
            and visible[j][0]+1<len(model)
            and model[visible[j][0]+1].startswith('<n:')for j in range(b,y))
        if kind!='replace' or max(z-a,y-b)>32 or unplaced_reference:
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
    # A damaged neighbor can prevent lexical alignment of several otherwise
    # separable words. Use unique complete physical recognitions as boundaries
    # inside that edit, never a positional zip or spelling similarity. These
    # anchors propose ownership only; each resulting edit still needs all the
    # ordinary transcription and uncertainty checks below.
    placed=[]
    for kind,a,z,b,y in operations:
        if kind!='replace' or max(z-a,y-b)<=1 or max(z-a,y-b)>32:
            placed.append((kind,a,z,b,y));continue
        def contextual(items):
            return any(len(own:=_whole_view_word(word,witness))==1
                and own[0].get('confidence',0)>=85
                and _key(own[0]['text'])==_key(word['text'])for word in items)
        if not(contextual(source[max(0,a-2):a])and contextual(source[z:z+2])):
            placed.append((kind,a,z,b,y));continue
        anchors=[]
        for i in range(a,z):
            own=_whole_view_word(source[i],witness)
            if len(own)!=1:continue
            box=own[0]['bbox']
            # A recognized complete word can contain several source fragments.
            # Anchoring just its stem would leave the same physical suffix as
            # another inserted word. Keep that edit intact for conservation.
            if any(k!=i and box[0]<=(source[k]['bbox'][0]+source[k]['bbox'][2])/2<=box[2]
                    and box[1]<=(source[k]['bbox'][1]+source[k]['bbox'][3])/2<=box[3]
                    for k in range(a,z)):continue
            matches=[j for j in range(b,y)if _key(own[0]['text'])==candidate[j]]
            if len(matches)==1:anchors.append((i,matches[0],id(own[0])))
        if (not anchors or len({j for i,j,w in anchors})!=len(anchors)
                or len({w for i,j,w in anchors})!=len(anchors)
                or [j for i,j,w in anchors]!=sorted(j for i,j,w in anchors)):
            placed.append((kind,a,z,b,y));continue
        left,right=a,b
        for i,j,_ in anchors:
            if left<i or right<j:
                placed.append(('replace'if left<i and right<j else 'delete'if left<i else 'insert',left,i,right,j))
            placed.append(('replace',i,i+1,j,j+1));left,right=i+1,j+1
        if left<z or right<y:
            placed.append(('replace'if left<z and right<y else 'delete'if left<z else 'insert',left,z,right,y))
    operations=placed
    for kind,a,z,b,y in operations:
        if source_print or a<z and all(proved_native_occurrence(w['text'],w['bbox'],native_print_words)for w in source[a:z]):
            # This occurrence is reliably decoded visible native paint, even
            # when a page uses the vision route for layout or other units.
            # Recognition settings cannot turn punctuation into words, remove
            # authored text, or change its spelling. The first reader's real
            # lexical edits remain recorded one word at a time.
            lexical=any(c.isalnum()for token in source_tokens[a:z]+proposed[b:y] for c in token)
            changed=kind!='equal' and lexical
            record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=source_tokens[a:z],
                status='RESOLVED',method='visible_mapped_native_glyphs',source_range=[a,z],model_range=[b,y])
            if changed:decisions.append(record)
            for j,word in enumerate(source[a:z]):
                note=(dict(record,source=[word['text']],chosen=[word['text']],
                    model=proposed[b+j:b+j+1] if z-a==y-b else []) if changed else None)
                chosen.append((word['text'],word['bbox'],note))
            if changed and a==z:changes.append((len(chosen),record,None))
            continue
        wrap=(_proved_wrap(source[a:z],proposed[b],witness,
                same_reading=bool(getattr(witness,'model_reading',False) and not unresolved_source
                    and not any(visible[i][0]in pending for i in range(b,y))))
              if kind!='equal'and y-b==1 else None)
        if wrap is not None:
            formatting.add(len(chosen));chosen.append((proposed[b],wrap,None));continue
        if (z-a==y-b==1 and source_tokens[a] in ('–','—') and proposed[b] in ('–','—')):
            own=[word for word in witness if _complete_word_occurrence(source[a],word)]
            if (len(own)==1 and own[0].get('confidence',0)>=90 and
                    own[0]['text'] in ('-','–','—')):
                # These recognizer codepoints describe the same complete
                # horizontal punctuation. Preserve the first reader's style;
                # no lexical word was corrected or remains in doubt.
                formatting.add(len(chosen));chosen.append((proposed[b],own[0]['bbox'],None));continue
        if kind=='equal':
            for token,index in zip(proposed[b:y],range(a,z)):
                own=[word for word in witness if _complete_word_occurrence(source[index],word)]
                word=own[0] if len(own)==1 else None
                if (token in ('–','—') and word and word.get('confidence',0)>=90
                        and word['text'] in ('-','–','—')):formatting.add(len(chosen))
                second=([other for other in corroboration if _same_occurrence(word,other)
                    and _key(word['text'])==_key(other['text'])] if word else [])
                from . import text_quality
                lexical=lambda value:''.join(c for c in _key(value).casefold() if c.isalpha())
                vocabulary=text_quality.english_lexicon()
                lexical_repair=(lexical(token) not in vocabulary and
                    not any(c.isupper() for c in token) and word is not None and
                    lexical(word['text']) in vocabulary)
                if (lexical(token) not in vocabulary and not any(c.isupper()for c in token)
                        and source[index].get('confidence',100)<85
                        and visible[b+index-a][0] in pending):
                    # A recovered low-confidence token can agree with the
                    # source OCR merely because it came from that same read.
                    # Consult its independent complete word, even when the
                    # broader contextual crop failed to recognize this word.
                    diverse=_own_word_choice(source[index],witness)
                    if (diverse and lexical(diverse['text']) in vocabulary
                            and not _small_mark_conflict(token,diverse['text'])
                            and not _joined_stroke_conflict(token,diverse['text'])):
                        word=diverse;lexical_repair=True;second=[diverse]
                corrected=bool(lexical_repair and word and word.get('confidence',0)>=90 and len(second)==1
                    and _key(word['text'])!=_key(token)
                    and _key(word['text']).rstrip('.,;:!?')!=_key(token).rstrip('.,;:!?')
                    and not _accent_conflict(word['text'],token))
                if corrected and hasattr(witness,'word_readings'):
                    diverse=_own_word_choice(source[index],witness)
                    corrected=bool(diverse and _key(diverse['text'])==_key(word['text']))
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
            own=[w for w in witness if _complete_word_occurrence(source[a],w)]
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
        # One fused optical token is not proof that a first reader invented a
        # printed space. Keep both literal words unless the pixels independently
        # establish their individual occurrences.
        fused_model=(z-a==1 and y-b>1 and
            placement(source_tokens[a])==''.join(placement(t)for t in proposed[b:y]))
        if fused_model:
            # Shared OCR segmentation cannot erase a plausible printed space.
            # Joining broken first-reader fragments needs positive independent
            # word networks and measured original ink at each proposed split.
            # A dictionary can veto merging complete words; it supplies no text.
            from .text_quality import english_lexicon
            parts=proposed[b:y];vocabulary=english_lexicon()
            separation=getattr(witness,'word_separation',None)
            measured=separation(source[a],tuple(parts))if separation is not None else None
            if (measured is True and getattr(witness,'model_reading',False)
                    and not getattr(witness,'failure',None) and not unresolved_source
                    and not any(visible[i][0]in pending for i in range(b,y))
                    and all(t.isascii()and t.isalpha()and t.casefold()in vocabulary for t in parts)
                    and ''.join(parts)==source_tokens[a]):
                # Individual original-ink runs prove the existing spaces.
                # No letter changes, and no neighboring lexical uncertainty,
                # arise merely from the source recognizer's fused token.
                for token in parts:
                    formatting.add(len(chosen));chosen.append((token,source[a]['bbox'],None))
                continue
            may_join=(not all(_key(t).casefold()in vocabulary for t in parts)
                      and measured is False)
            joined=_own_word_choice(source[a],witness)if may_join else None
            supported=([('word',[joined['text']],[joined])]
                if joined and _key(joined['text'])==_key(source_tokens[a])else [])
        incomplete_ink=(z-a==y-b==1 and source[a].get('confidence',0)<85
            and any(c.isalpha()for c in proposed[b])
            and not source_tokens[a].endswith(('-', '\u00ad')) and any(
            _same_occurrence(source[a],w) and w.get('confidence',0)>=85
            and min(source[a]['bbox'][3],w['bbox'][3])-max(source[a]['bbox'][1],w['bbox'][1])
                <.55*(source[a]['bbox'][3]-source[a]['bbox'][1])
            for w in witness))
        if incomplete_ink:supported=[]
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
        # Letter noise resembling a number is not a verified replacement of a
        # well-formed ordinal. Inspect that exact original word with separate
        # language networks; if they cannot settle it, publish original ink.
        ordinal_conflict=(z-a==y-b==1 and _ordinal(proposed[b]) and any(
            len(choice[1])==1 and not _ordinal(choice[1][0]) and _ordinal_noise(choice[1][0])
            for choice in supported))
        if ordinal_conflict:supported=[]
        lexical_conflict=(z-a==y-b==1 and len(supported)==1
            and len(supported[0][1])==1
            and _dictionary_regression(proposed[b],supported[0][1][0]))
        stroke_conflict=(z-a==y-b==1 and len(supported)==1
            and len(supported[0][1])==1
            and _joined_stroke_conflict(proposed[b],supported[0][1][0]))
        if lexical_conflict or stroke_conflict:supported=[]
        # A page read and a differently cropped read of the same English
        # network can repeat a confident error, including another dictionary
        # word. The production witness supplies separate English/Latin word
        # networks and Latin contextual frames. Require that diverse evidence
        # before overriding the first reader's lexical word. An explicit
        # unavailable second network cannot silently reuse the first one.
        network_conflict=(z-a==y-b==1 and len(supported)==1
            and len(supported[0][1])==1 and hasattr(witness,'word_readings')
            and _letter_number_identity(supported[0][1][0])!=_letter_number_identity(proposed[b])
            and any(c.isalpha()for c in proposed[b]+supported[0][1][0]))
        if network_conflict:supported=[]
        alternate=None;resampled=None;word_choice=None;word_method='two_own_word_language_readings'
        if len(supported)!=1 and z-a==y-b==1 and not partial_wrap:
            word_choice=_own_word_choice(source[a],witness)
            if word_choice is None:
                word_choice=_model_word_choice(source[a],witness,proposed[b])
                word_method='intact_pixel_corroboration_of_model'
            if word_choice and _accent_conflict(word_choice['text'],proposed[b]):word_choice=None
            if word_choice and max(SequenceMatcher(None,placement(word_choice['text']),placement(t)).ratio()
                                   for t in (source_tokens[a],proposed[b]))<.65:word_choice=None
            if word_choice:supported=[('word',[word_choice['text']],[word_choice])]
            else:resampled=_resampled_model_choice(source,a,witness,proposed[b])
            if resampled:supported=[('resampled',[resampled['text']],[resampled])]
            if not word_choice and not resampled:
                alternate=_alternate_choice(source,a,witness,proposed[b])
                if alternate:supported=[('alternate',[alternate['text']],[alternate])]
        # Separate language networks can share an optical error, including a
        # complete-looking nonword. A later crop cannot bypass the contextual
        # witness's safety veto. This never selects the dictionary spelling:
        # retain original ink, including a possible unusual printed spelling.
        if z-a==y-b==1 and len(supported)==1 and len(supported[0][1])==1:
            unsafe_lexical=_dictionary_regression(proposed[b],supported[0][1][0])
            unsafe_stroke=_joined_stroke_conflict(proposed[b],supported[0][1][0])
            if unsafe_lexical or unsafe_stroke:
                lexical_conflict=lexical_conflict or unsafe_lexical
                stroke_conflict=stroke_conflict or unsafe_stroke
                supported=[];word_choice=None;resampled=None;alternate=None
        if ordinal_conflict:supported=[choice for choice in supported if len(choice[1])==1 and _ordinal(choice[1][0])]
        quote_conflict=(z-a==y-b==1 and _quote_conflict(source_tokens[a],proposed[b])
            and (len(supported)!=1 or len(supported[0][1])!=1
                 or _quote_conflict(supported[0][1][0],proposed[b])
                 or any(not c.isalnum()and not c.isspace()
                        and c not in _key(source_tokens[a])+_key(proposed[b])
                        for c in _key(supported[0][1][0]))))
        initialism_conflict=(z-a==y-b==1 and _initialism_mark_conflict(
            source_tokens[a],proposed[b],supported[0][1][0]
            if len(supported)==1 and len(supported[0][1])==1 else None))
        if initialism_conflict:supported=[];boundary=None
        if (z-a==y-b==1 and len(supported)==1 and len(supported[0][1])==1
                and any((c in ('"',"'")or unicodedata.category(c)in('Pi','Pf'))
                        and c not in _key(source_tokens[a])+_key(proposed[b])
                        for c in _key(supported[0][1][0]))):
            # A late alphabet candidate can turn a damaged capital stroke into
            # a novel quote. Compare its punctuation after every candidate path,
            # even when neither original reading contained that invented mark.
            quote_conflict=True
        if quote_conflict:
            # Agreement between OCR settings can repeat a missing apostrophe or
            # collapse the first reading's double quote. It cannot authorize
            # that change. Agreement with the first reading's quote identity,
            # however, can reject an extra quote invented by the source OCR.
            supported=[];boundary=None
        word_scope_conflict=False;word_scope=None
        scope_reader=getattr(witness,'word_scope',None)
        if (scope_reader is not None and z-a==y-b==1 and len(supported)==1
                and len(supported[0][1])==1):
            first=_letter_number_identity(proposed[b]).casefold()
            settled=_letter_number_identity(supported[0][1][0]).casefold()
            shortening=(len(first)>len(settled)>=2 and
                        (first.startswith(settled)or first.endswith(settled)))
            if shortening:
                word_scope=scope_reader(source[a])
                word_scope_conflict=not _covers_ink_scope(supported[0][2],word_scope)
                if word_scope_conflict:supported=[];boundary=None
        symbol_conflict=(z-a==y-b==1 and _symbol_identity_conflict(
            proposed[b],source_tokens[a]))
        if symbol_conflict:
            # Neither alphabetic network is a mathematical-glyph classifier.
            # Keep the exact original ink rather than certify its digit guess.
            supported=[];boundary=None
        partial_counter=word_scope_conflict and word_scope is not None
        retained_vision=(len(supported)!=1 and z-a==y-b==1
            and getattr(witness,'model_reading',False) and not unresolved_source
            and visible[b][0]not in pending and (source[a].get('confidence',0)<85 or partial_counter)
            and not(ordinal_conflict or quote_conflict or symbol_conflict or stroke_conflict or initialism_conflict or incomplete_ink or partial_wrap or word_scope_conflict and not partial_counter
                    or _small_mark_conflict(source_tokens[a],proposed[b]))
            and not _optical_counterread(source[a],witness,proposed[b],word_scope if partial_counter else None))
        if retained_vision:
            evidence=dict(source[a],text=proposed[b])
            if partial_counter:
                own=source[a]['bbox'];evidence['bbox']=[min(own[0],word_scope[0]),min(own[1],word_scope[1]),max(own[2],word_scope[2]),max(own[3],word_scope[3])]
            supported=[('model',proposed[b:y],[evidence])]
        resolved=len(supported)==1
        unproved_multiword=(not resolved and bool(source_tokens[a:z]) and bool(proposed[b:y])
            and getattr(witness,'model_reading',False) and not unresolved_source
            and not native_print_words and not(fused_model or wrapped or partial_wrap)
            and max(z-a,y-b)>1 and boundary is None
            and ''.join(placement(t)for t in source_tokens[a:z])!=
                ''.join(placement(t)for t in proposed[b:y]))
        name,literal,evidence=(supported[0] if resolved else
            boundary if boundary else ('source',source_tokens[a:z],[]) if unresolved_source or unproved_multiword or not proposed[b:y] else ('model',proposed[b:y],[]))
        record=dict(source=source_tokens[a:z],model=proposed[b:y],chosen=list(literal),
                    status='RESOLVED' if resolved else 'UNRESOLVED',
                    method=('complete_vision_reading_without_optical_counterevidence'if retained_vision else word_method if word_choice else 'corroborated_resampled_pixel_readings' if resampled else 'two_alternate_alphabet_pixel_readings' if alternate else 'independent_upright_crop') if resolved else 'pixel_word_boundary_only' if boundary else 'pixel_read_inconclusive',
                    source_range=[a,z],model_range=[b,y])
        if (ordinal_conflict or quote_conflict or lexical_conflict or stroke_conflict or initialism_conflict or network_conflict or symbol_conflict) and not resolved:
            record.update(source_image_only=True,source_bbox=list(source[a]['bbox']))
            if quote_conflict:record['method']='unproved_quote_identity'
            elif initialism_conflict:record['method']='unproved_initialism_punctuation'
            elif stroke_conflict:record['method']='unproved_shared_stroke_identity'
            elif lexical_conflict:record['method']='unproved_shared_network_nonword'
            elif symbol_conflict:record['method']='unproved_mathematical_symbol_identity'
            elif network_conflict:record['method']='unproved_shared_network_word'
        if word_scope_conflict and not resolved:
            record.update(source_image_only=True,
                source_bbox=list(word_scope or source[a]['bbox']),
                method='unproved_complete_word_scope')
        if incomplete_ink and not resolved:
            record.update(source_image_only=True,source_bbox=list(source[a]['bbox']),
                method='incomplete_vertical_word_ink')
        unproved_source_choice=(not resolved and bool(literal)
            and list(literal)==source_tokens[a:z]and list(literal)!=proposed[b:y])
        withheld=set()
        if unproved_source_choice:
            # An admitted local reading remains available when a later crop
            # merely cannot repeat it. Withhold the specific weak occurrence
            # or a complete competing read, not its whole conserved passage.
            # Symbols remain available for the separate note-origin verifier.
            for j,word in enumerate(source[a:z]):
                weak=word.get('confidence')is not None and word['confidence']<85
                counter=any(_complete_word_occurrence(word,other)
                    and other.get('confidence',0)>=85
                    and _key(other['text'])!=_key(word['text'])for other in witness)
                # Two source records claiming the same physical word are not
                # independent occurrences, even if their confidence is absent
                # or high. A hidden layer can append several guessed labels at
                # one interpolated position; conservation cannot print them.
                collision=any(index!=a+j and all(abs(x-y)<.1
                    for x,y in zip(word['bbox'],other['bbox']))
                    for index,other in enumerate(source))
                if any(c.isalnum()for c in word['text'])and(weak or counter or collision or unproved_multiword):withheld.add(j)
            if withheld:record['method']='unproved_source_transcription_change'
        decisions.append(record)
        boxes=([w['bbox'] for w in evidence] if evidence else
               [w['bbox'] for w in source[a:z]])
        box=( [min(x[0] for x in boxes),min(x[1] for x in boxes),
               max(x[2] for x in boxes),max(x[3] for x in boxes)] if boxes else None)
        single_ink_word=(len(literal)==len(evidence)==1 and z-a>1 and
            _covers_ink_scope(evidence,[min(w['bbox'][0]for w in source[a:z]),
                min(w['bbox'][1]for w in source[a:z]),max(w['bbox'][2]for w in source[a:z]),
                max(w['bbox'][3]for w in source[a:z])]))
        for j,token in enumerate(literal):
            # Several source fragments can belong to this one positively
            # located word. Keep their transcription evidence on that word,
            # rather than report a fabricated positional or unplaced mapping.
            word_record=record if single_ink_word else _word_note(record,j,len(decisions)-1)
            if j in withheld:
                own=list(source[a+j]['bbox'])
                word_record=dict(word_record,source_image_only=True,
                    source=[source[a+j]['text']],source_bbox=own,
                    source_range=[a+j,a+j+1])
                word_record.pop('mapping',None)
                chosen.append((token,own,word_record))
            else:chosen.append((token,evidence[j]['bbox'] if evidence else box,word_record))
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
        if source_print:continue
        affected=([i for j,i in zip(range(a,z),range(b,y)) if visible[j][0] in pending]
            if kind=='equal' else list(range(b,y)) if any(visible[j][0] in pending for j in range(a,z)) else [])
        for i in affected:
            if i in formatting:
                verified.append(i);continue
            if proved_native_occurrence(tokens[i],bounds.get(i),native_print_words):continue
            # A reader's low-confidence flag is not permanent uncertainty when
            # the same occurrence has an exact, uniquely contextual pixel read.
            left=max(0,i-1);right=min(len(tokens),i+2)
            locations=_matches(pixels,[_key(t) for t in tokens[left:right]])
            context=witness[locations[0]:locations[0]+right-left] if len(locations)==1 else []
            own=[word for word in witness if i in bounds and
                 _complete_word_occurrence(dict(bbox=bounds[i]),word)]
            own_exact=(len(own)==1 and own[0].get('confidence',0)>=90
                       and _key(own[0]['text'])==_key(tokens[i]))
            source_own=[word for word in source if i in bounds and
                        _complete_word_occurrence(dict(bbox=bounds[i]),word)]
            source_exact=(not getattr(witness,'failure',None) and len(source_own)==1 and source_own[0].get('confidence',0)>=90
                          and _key(source_own[0]['text'])==_key(tokens[i])
                          and not any(w.get('confidence',0)>=85 and _key(w['text'])!=_key(tokens[i])for w in own))
            if i not in notes and (own_exact or source_exact):
                verified.append(i);continue
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
        result=CropWitness(_read_crop(display, regions, cache))
        # Recognition remains local and lazy. Ordinary settled English edits
        # incur no extra passes. The larger alphabet is used only to adjudicate
        # an unresolved single-word edit at its own physical occurrence.
        from functools import lru_cache
        @lru_cache(maxsize=1)
        def alternatives():
            try:
                import shutil
                engine=shutil.which('tesseract')
                directory=_latin_directory(engine) if engine else None
                if not directory:return []
                heights=[r[3]-r[1]for r in regions if r[3]>r[1]]
                # Tiny printed letters need enough samples to distinguish their
                # strokes. Both frames remain bounded by four million pixels.
                base=min(8,max(4,24/min(heights))) if heights else 4
                frames=[regions]
                contextual=alternate_regions(regions)
                if contextual!=[list(r)for r in regions]:frames.append(contextual)
                return [_read_crop(display,frame,cache,language='LatinRaw',scale=scale,tessdata=directory)
                        for frame in frames for scale in (base,base*1.25)]
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        @lru_cache(maxsize=1)
        def resampled():
            # Upscaling a small raster can reduce recognition confidence.
            # Two smaller frames may corroborate the model's own spelling;
            # neither frame alone can certify it or introduce another word.
            try:
                return [_read_crop(display,regions,cache,scale=scale) for scale in (1.5,2)]
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        result.resampled=resampled
        result.alternates=alternatives
        @lru_cache(maxsize=128)
        def own_word(bounds):
            try:
                import shutil
                engine=shutil.which('tesseract')
                directory=_latin_directory(engine) if engine else None
                if not directory:return []
                width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
                if not (0<width<=200 and 0<height<=80 and width/height<=18):return []
                networks=[('eng',None),('LatinRaw',directory)]
                views=[_read_crop(display,[bounds],cache,language=language,scale=6,
                                  tessdata=data,psm=8)for language,data in networks]
                if _paired_word_choice(dict(bbox=bounds),views):return views
                # A recognizer's single-word mode can assign zero confidence
                # to legible ink. Single-line mode has a separate segmentation
                # path; it still needs both trained networks on the complete
                # same occurrence. Conflicting positive frame pairs cannot win.
                views.extend(_read_crop(display,[bounds],cache,language=language,
                    scale=scale,tessdata=data,psm=7)
                    for scale in (3,4,6)for language,data in networks)
                return views
            except (RuntimeError,OSError,ValueError,KeyError,subprocess.SubprocessError):return []
        @lru_cache(maxsize=128)
        def word_scope(bounds):
            try:
                limits=_neighbor_line_limits(bounds,result)
                return _word_ink_scope(display,dict(bbox=bounds),line_limits=limits)
            except (RuntimeError,OSError,ValueError,KeyError):return None
        @lru_cache(maxsize=128)
        def word_separation(bounds,parts):
            try:return _word_separation(display,dict(bbox=bounds),parts)
            except (RuntimeError,OSError,ValueError,KeyError):return None
        result.word_separation=lambda word,parts:word_separation(tuple(word['bbox']),tuple(parts))
        result.word_scope=lambda word:word_scope(tuple(word['bbox']))
        result.word_readings=lambda word:own_word(tuple(word['bbox']))
        result.mark_absence=lambda word:below_dot_absent(display,word)
        return result
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result = CropWitness()
        result.failure = f'{type(error).__name__}: {error}'
        return result


def _word_separation(display,word,parts):
    """Measure a proposed space only when individual ink runs are unambiguous.

    Exact Latin letter counts bind component-column runs to the proposed split.
    Touching/broken characters, punctuation and ambiguous gap sizes decline the
    proof. The dictionary never supplies the joined spelling.
    """
    from . import extract,vision_pixels
    if not(2<=len(parts)<=3 and all(p.isascii()and p.isalpha()for p in parts)):return None
    a=word['bbox'];height=a[3]-a[1]
    if not(0<height<=80 and 0<a[2]-a[0]<=200):return None
    box=extract.pymupdf.Rect(a)&display.rect
    pix=display.pixmap(box,scale=8,max_pixels=200_000)
    picture,components=vision_pixels._word_ink_components(pix)
    runs=[]
    for left,right in sorted((c[0],c[2])for c in components):
        if runs and left<=runs[-1][1]:runs[-1][1]=max(runs[-1][1],right)
        else:runs.append([left,right])
    if len(runs)!=sum(map(len,parts)):return None
    cuts=[];at=0
    for part in parts[:-1]:
        at+=len(part)
        cuts.append((runs[at][0]-runs[at-1][1])*box.width/picture.width)
    if all(gap>=height*.25 for gap in cuts):return True
    if all(gap<=height*.15 for gap in cuts):return False
    return None


def _neighbor_line_limits(bounds,words):
    """Separate nearby rows using two independently located words per row.

    These positions only bound the ink search. They neither choose a spelling
    nor let the recognizer's own shortened word box certify a cropped prefix.
    No reliable neighboring row means the original larger search stays in use.
    """
    a=bounds;height=a[3]-a[1]
    if height<=0:return None,None
    neighbors=[w['bbox']for w in words if w.get('confidence',0)>=85
        and w['bbox'][3]-w['bbox'][1]>=.4*height
        and w['bbox'][0]<a[2]+2*height and w['bbox'][2]>a[0]-2*height]
    above=[b for b in neighbors if b[3]<a[1]]
    below=[b for b in neighbors if b[1]>a[3]]
    top=bottom=None
    if above:
        edge=max(b[3]for b in above)
        row=[b for b in above if edge-b[3]<=.5*height]
        if len({tuple(b)for b in row})>=2:top=(edge+a[1])/2
    if below:
        edge=min(b[1]for b in below)
        row=[b for b in below if b[1]-edge<=.5*height]
        if len({tuple(b)for b in row})>=2:bottom=(edge+a[3])/2
    return top,bottom


def _word_ink_scope(display,word,*,line_limits=(None,None)):
    """Bound the adjacent original ink before certifying a shortened word.

    The recognizer's own box can already exclude a prefix. Measure a bounded
    surrounding raster instead; nearby connected ink is evidence against that
    box, never authority for a different spelling. Ambiguous/clipped territory
    declines the proof rather than treating repeated tail recognition as whole.
    """
    from . import extract,vision_pixels
    a=word['bbox'];height=a[3]-a[1];width=a[2]-a[0]
    if not(0<height<=80 and 0<width<=200):return None
    bounds=extract.pymupdf.Rect(a[0]-2*height,a[1]-.75*height,
                               a[2]+2*height,a[3]+.75*height)&display.rect
    top,bottom=line_limits
    if top is not None:bounds.y0=max(bounds.y0,top)
    if bottom is not None:bounds.y1=min(bounds.y1,bottom)
    pix=display.pixmap(bounds,scale=6,max_pixels=200_000)
    picture,components=vision_pixels._word_ink_components(pix)
    if not components:return None
    sx=bounds.width/picture.width;sy=bounds.height/picture.height
    boxes=[[bounds.x0+c[0]*sx,bounds.y0+c[1]*sy,
            bounds.x0+c[2]*sx,bounds.y0+c[3]*sy]for c in components]
    selected=[b for b in boxes if min(a[2],b[2])>max(a[0],b[0])
              and min(a[3],b[3])>max(a[1],b[1])]
    if not selected:return None
    # Font metrics can exceed the visible letters, while a cropped OCR box
    # can exclude their ascenders. Use the complete selected ink envelope,
    # including detached dots, rather than either misleading height.
    left=min(b[0]for b in selected);right=max(b[2]for b in selected)
    for _ in range(len(boxes)):
        ink_height=max(b[3]for b in selected)-min(b[1]for b in selected)
        gap=max(.5,.32*ink_height)
        adjacent=[b for b in boxes if b[0]<=right+gap
                  and b[2]>=left-gap]
        new_left=min(b[0]for b in adjacent);new_right=max(b[2]for b in adjacent)
        if new_left==left and new_right==right:break
        left,right=new_left,new_right;selected=adjacent
    if left<=bounds.x0+sx or right>=bounds.x1-sx:return None
    if right-left>2.5*width:return None
    ink=[b for b in boxes if b[0]<=right and b[2]>=left]
    if top is not None and min(b[1]for b in ink)<=bounds.y0+sy:return None
    if bottom is not None and max(b[3]for b in ink)>=bounds.y1-sy:return None
    return [left,min(b[1]for b in ink),right,max(b[3]for b in ink)]


def _covers_ink_scope(evidence,scope):
    if not evidence or scope is None:return False
    boxes=[w['bbox']for w in evidence]
    box=[min(b[0]for b in boxes),min(b[1]for b in boxes),
         max(b[2]for b in boxes),max(b[3]for b in boxes)]
    return (min(box[2],scope[2])-max(box[0],scope[0])>=.9*(scope[2]-scope[0])
            and min(box[3],scope[3])-max(box[1],scope[1])>=.9*(scope[3]-scope[1]))


def _read_crop(display, regions, cache, language='eng', scale=4, tessdata=None,psm=6):
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
        pix=display.pixmap(box,scale=scale,max_pixels=4_000_000)
        image=pix.tobytes('png');border=0
        if language=='LatinRaw':
            from PIL import Image,ImageOps
            border=16
            picture=ImageOps.expand(Image.open(io.BytesIO(image)).convert('RGB'),border=border,fill='white')
            encoded=io.BytesIO();picture.save(encoded,format='PNG');image=encoded.getvalue()
        model_identity=(_latin_digest() if language=='LatinRaw' else b'')
        identity=hashlib.sha256(f'upright-block-{language}-psm{psm}-scale{scale}-v3\0'.encode()+model_identity+_engine_identity(engine)+image).hexdigest()
        target=root/(identity+'.json')
        with (root/(identity+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists():rows=json.loads(target.read_text())
            else:
                command=[engine,'stdin','stdout','-l',language,'--psm',str(psm)]
                if tessdata:command+=['--tessdata-dir',str(tessdata)]
                command+=['-c','tessedit_create_tsv=1']
                result=subprocess.run(command,
                    input=image,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,
                    env=dict(os.environ,OMP_THREAD_LIMIT='1'))
                if result.returncode:raise RuntimeError('Independent crop recognition failed')
                rows=[]
                for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')),
                    delimiter='\t',quoting=csv.QUOTE_NONE):
                    if row['level']!='5' or not row['text'].strip():continue
                    x,y,w,h=[int(row[k]) for k in ('left','top','width','height')]
                    rows.append(dict(text=row['text'],confidence=float(row['conf']),
                        bbox=[(x-border)/pix.width,(y-border)/pix.height,(x+w-border)/pix.width,(y+h-border)/pix.height]))
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
    output=[];notes={};bounds={};decisions=[];gaps=[]
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
        if kind=='insert' and not resolved and unique and anchored and a-1 in source_map and a in source_map:
            left_box=source_map[a-1]['bbox'];right_box=source_map[a]['bbox']
            height=max(left_box[3]-left_box[1],right_box[3]-right_box[1],1)
            # Additional recognized ink is not a proved letter. When it sits
            # inside this exact short same-row gap, retain the gap and following
            # word as original ink rather than silently dropping the edit.
            # A remote row or another column cannot own it.
            def in_gap(word):
                box=word['bbox'];small=min(box[3]-box[1],left_box[3]-left_box[1],right_box[3]-right_box[1])
                return (word.get('confidence',0)>=40 and any(c.isalnum()or c=='|'for c in word['text'])
                    and box[0]>=left_box[2]-.5 and box[2]<=right_box[0]+.5
                    and 0<right_box[0]-left_box[2]<=height*4
                    and min(box[3],left_box[3],right_box[3])-max(box[1],left_box[1],right_box[1])>=small*.5)
            if evidence and all(in_gap(word)for word in evidence):
                boxes=[right_box]+[word['bbox']for word in evidence]
                gaps.append((len(output),[min(b[0]for b in boxes),min(b[1]for b in boxes),
                    max(b[2]for b in boxes),max(b[3]for b in boxes)]))
        chosen=[w['text'] for w in evidence] if resolved else tokens[a:z]
        record=dict(source=tokens[a:z],model=tokens[a:z],chosen=chosen,
            status='RESOLVED' if resolved else 'UNRESOLVED',
            method=('two_local_pixel_readings' if two_views else 'pixel_case_agreement' if case_only
                    else 'contextual_crop_lexical_support') if resolved else 'pixel_read_inconclusive')
        decisions.append(record)
        for i,t in enumerate(chosen):
            index=len(output);output.append(t);notes[index]=_word_note(record,i,len(decisions)-1)
            if resolved:bounds[index]=evidence[i]['bbox']
            elif a+i in source_map:bounds[index]=source_map[a+i]['bbox']
    for index,box in gaps:
        if index>=len(output):continue
        record=dict(source=[output[index]],model=[output[index]],chosen=[output[index]],
            status='UNRESOLVED',method='unsettled_additional_pixel_ink',
            source_image_only=True,source_bbox=box)
        notes[index]=record;decisions.append(record)
    return dict(tokens=output,boxes=bounds,marked=set(notes),word_notes=notes,decisions=decisions)


def audit_agreement(tokens, source, witness, boundary_proof=None, word_boxes=None):
    """Flag pixel conflicts even where a layer and the first reading agree.

    A recognizer's different spelling is evidence of a conflict, not authority
    to replace a valid source word. Only the source occurrence's own overlapping
    ink can cause a mark; a different copy of a repeated word cannot do so.
    """
    visible=[(i,token)for i,token in enumerate(tokens) if not token.startswith('<n:')]
    if word_boxes is None:
        if len(visible)!=len(source):return {}
        occurrences={i:word for (i,token),word in zip(visible,source)}
    else:
        # Opposite word-boundary errors can cancel in total count. The lexical
        # adjudicator has already bound each chosen word to its own occurrence;
        # positional zipping would move every intervening word to its neighbour.
        # A shared span box is not a uniquely placed word and earns no new claim.
        counts={}
        for i,token in visible:
            if i in word_boxes:
                key=tuple(word_boxes[i]);counts[key]=counts.get(key,0)+1
        occurrences={i:dict(text=token,bbox=word_boxes[i]) for i,token in visible
                     if i in word_boxes and counts[tuple(word_boxes[i])]==1}
    # Recognition may fuse adjacent printed words. Exact letters plus an
    # independently observed blank between every original occurrence can prove
    # the existing spaces. Geometry alone and a dictionary never waive a clash.
    segmented=set();fused={}
    for other in witness:
        owned=[(i,w)for i,w in occurrences.items()if _same_occurrence(w,other)]
        if not 2<=len(owned)<=3 or other.get('confidence',0)<85:continue
        if ''.join(_key(w['text'])for j,w in owned)!=_key(other['text']):continue
        for j,w in owned:fused.setdefault(j,set()).add(id(other))
        if boundary_proof is not None and all(
                boundary_proof(a,b)for (_,a),(_,b)in zip(owned,owned[1:])):
            segmented.update(j for j,w in owned)
    notes={}
    for i,token in visible:
        if i in segmented or i not in occurrences:continue
        word=occurrences[i]
        # A fused recognition can contradict several complete source words,
        # but a narrow glyph fragment cannot contradict its whole owner.
        overlapping=[w for w in witness if _same_occurrence(word,w) and
            (id(w)in fused.get(i,set()) or
             _complete_recognition(word,w))]
        if len(overlapping)!=1:continue
        other=overlapping[0]
        if other.get('confidence',0)<85 or _key(other['text'])==_key(token):continue
        if (token in ('–','—') and other['text'] in ('-','–','—') and
                _complete_word_occurrence(word,other)):continue
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


def replace_formatted_word(fragment, chosen):
    """Replace text characters while preserving their inline style boundaries.

    Typed references, ordinals and other structural elements are deliberately
    outside this lexical edit. Character alignment maps existing style cuts;
    it never moves an action or strips its tags to obtain a spelling.
    """
    import re
    from html import escape
    from . import annotate
    if any(name.casefold() not in ('em','strong','i','b','span')
           for name in re.findall(r'</?([a-zA-Z][\w:-]*)\b',fragment)):
        return None
    nodes=[n for n in annotate._text_nodes(fragment,include_uncertain=True)if n.plain]
    original=''.join(n.plain for n in nodes)
    if not original or any(c.isspace()for c in original+chosen):return None
    def boundary(offset):
        if offset==len(original):return len(chosen)
        for _,a,z,b,y in SequenceMatcher(None,original,chosen,autojunk=False).get_opcodes():
            if a<=offset<z:return b+round((offset-a)*(y-b)/(z-a))
        return 0
    edits=[];offset=0
    for node in nodes:
        end=offset+len(node.plain)
        edits.append((node.starts[0],node.ends[-1],escape(chosen[boundary(offset):boundary(end)])))
        offset=end
    for a,z,value in reversed(edits):fragment=fragment[:a]+value+fragment[z:]
    return fragment



def _latin_digest():
    return bytes.fromhex('28acb38d9613595a7ae88a427bc7481c2ee488b960ea6baa31945f5ce0248b61')


def _latin_directory(engine):
    """Find only the hash-verified optical data installed at image build time."""
    import os
    return _cached_latin_directory(engine,os.environ.get('REFLOW_LATIN_TESSDATA',''),os.environ.get('TESSDATA_PREFIX',''))


@lru_cache(maxsize=8)
def _cached_latin_directory(engine,alternate_root,primary_root):
    import re,subprocess
    listing=subprocess.run([engine,'--list-langs'],capture_output=True,timeout=5,check=True)
    roots=[alternate_root,primary_root]
    roots+=re.findall(r'"([^"\n]+)"',(listing.stdout+listing.stderr).decode(errors='replace'))
    return _verified_latin_directory(tuple(roots))


@lru_cache(maxsize=8)
def _verified_latin_directory(roots):
    import hashlib
    from pathlib import Path
    for root in roots:
        if not root:continue
        for path in (Path(root)/'LatinRaw.traineddata',Path(root)/'tessdata/LatinRaw.traineddata'):
            if path.is_file() and hashlib.sha256(path.read_bytes()).digest()==_latin_digest():return path.parent
    return None


def _resampled_model_choice(source,index,witness,model_word):
    """Corroborate one model spelling without trusting an enlarged OCR guess.

    The primary crop must locate the same occurrence without a confident
    conflicting spelling. Its weak spelling need not repeat the model's.
    Two additional bounded frames must independently print it with confidence
    and a common neighboring anchor. English cannot establish accent changes.
    The caller retains the correction's source-linked mark after settlement.
    """
    reader=getattr(witness,'resampled',None)
    if reader is None:return None
    if any(c.isalpha() and not c.isascii() for c in source[index]['text']+model_word):return None
    own=[w for w in witness if _complete_word_occurrence(source[index],w)]
    if len(own)!=1:return None
    if (_key(own[0]['text'])!=_key(model_word)
            and own[0].get('confidence',0)>=85):return None
    choice=_alternate_pair_choice(source,index,witness,model_word,reader())
    if choice is None or _key(choice['text'])!=_key(model_word):return None
    return choice


def _alternate_choice(source,index,witness,model_word):
    reader=getattr(witness,'alternates',None)
    if reader is None:return None
    views=reader();choices={}
    for start in range(0,len(views),2):
        choice=_alternate_pair_choice(source,index,witness,model_word,views[start:start+2])
        if choice:choices[_key(choice['text'])]=choice
    # Two incompatible confident frame pairs cannot settle one printed word.
    return next(iter(choices.values()))if len(choices)==1 else None


def _alternate_pair_choice(source,index,witness,model_word,views):
    """Require two confident optical reads and a uniquely anchored occurrence.

    The script network includes macron vowels but no combining under-dot. Its
    omission cannot certify removing that mark from a first reading. English
    OCR is still never allowed to certify accent presence or absence.
    """
    if sum(c.isalnum()for c in source[index]['text'])<2:return None
    if len(views)!=2:return None
    own=[[w for w in view if _complete_word_occurrence(source[index],w)]for view in views]
    if not all(len(row)==1 and row[0].get('confidence',0)>=85 for row in own):return None
    first,second=own[0][0],own[1][0]
    if _key(first['text'])!=_key(second['text']):return None
    bare=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())if c.isalnum())
    if max(SequenceMatcher(None,bare(first['text']),bare(t)).ratio()
           for t in (source[index]['text'],model_word))<.65:return None
    marks=lambda value:{c for c in unicodedata.normalize('NFD',value)if unicodedata.combining(c)}
    if '\u0323'in marks(model_word)-marks(first['text']):
        proof=getattr(witness,'mark_absence',None)
        if proof is None or not proof(first):return None
    neighbors=source[max(0,index-2):index]+source[index+1:index+3]
    if not any(all(len([w for w in view if _same_occurrence(n,w)
            and _key(n['text']).casefold()==_key(w['text']).casefold()
            and w.get('confidence',0)>=85])==1 for view in views)for n in neighbors):return None
    return first


def disclose_alternate_conflicts(result, boxes, witness, allow_clear=True):
    """Retain the original ink for an already unresolved optical disagreement.

    A low-confidence alternate is evidence of ambiguity, not authority to print
    its guess. It can request a source-word image only at one physical word and
    only when neither alternate layout supports the live claim. Their own
    disagreement cannot select either spelling; it also requires original ink.
    """
    reader=getattr(witness,'alternates',None)
    if reader is None:return
    pending=[i for i in result['marked']if i in boxes
        and not result['tokens'][i].startswith('<n:')
        and (result.get('word_notes')or{}).get(i,{}).get('status')!='RESOLVED'
        and not (result.get('word_notes')or{}).get(i,{}).get('source_image_only')]
    if not pending:return
    views=reader()[:2]
    if len(views)!=2:return
    for index in pending:
        original=dict(bbox=boxes[index])
        token=result['tokens'][index];note=(result.get('word_notes')or{}).get(index,{})
        # A contextual network can misread one intact word even when its own
        # crop corroborates the unchanged source and page reading. Ask that
        # occurrence before turning noisy alternate spellings into an image.
        # Corrections retain their provenance; local-only OCR stays unverified.
        from .text_quality import english_lexicon
        lexical=''.join(c for c in _key(token).casefold()if c.isalpha())
        unchanged=note.get('source')==note.get('model')==note.get('chosen')==[token]
        paired=(_own_word_choice(original,witness)
            if allow_clear and unchanged and len(lexical)>=3 else None)
        exact_pair=(paired is not None and _key(paired['text'])==_key(token)
                    and paired.get('confidence',0)>=90)
        if (allow_clear and unchanged and (exact_pair or lexical in english_lexicon()
                and _model_word_choice(original,witness,token) is not None)):
            result['marked'].discard(index);result['word_notes'].pop(index)
            result.setdefault('verified_unchanged',[]).append(index)
            result['adjudication']=[d for d in result.get('adjudication',[])if d is not note]
            continue
        own=[_whole_view_word(original,view)for view in views]
        if not all(len(row)==1 for row in own):continue
        a,b=own[0][0],own[1][0]
        if (allow_clear and all(_key(w['text'])==_key(token)and w.get('confidence',0)>=90 for w in (a,b))
                and note.get('source')==note.get('model')==note.get('chosen')==[token]):
            result['marked'].discard(index);result['word_notes'].pop(index)
            result.setdefault('verified_unchanged',[]).append(index)
            result['adjudication']=[d for d in result.get('adjudication',[])if d is not note]
            continue
        # A different local alphabet does not invalidate positive independent
        # evidence for this occurrence. Keep its original uncertainty, rather
        # than replacing supported prose with the alternate network's noise.
        own_primary=[w for w in witness if _complete_word_occurrence(original,w)]
        letters=lambda value:''.join(c for c in _key(value).casefold()if c.isalpha())
        lexical=letters(token)
        from .text_quality import english_lexicon
        if (len(own_primary)==1 and own_primary[0].get('confidence',0)>=85
                and lexical in english_lexicon()
                and lexical and letters(own_primary[0]['text'])==lexical):continue
        if (note.get('source')==note.get('chosen')==[token]
                and len(own_primary)==1 and lexical
                and letters(own_primary[0]['text'])==lexical):continue
        original_words=[w for w in getattr(witness,'source_occurrences',())
                        if _complete_word_occurrence(original,w)]
        if (len(original_words)==1 and original_words[0].get('confidence',0)>=85
                and lexical and letters(original_words[0]['text'])==lexical):
            # The original occurrence still supplies a supported alternative.
            # Do not resolve it: retain its marked, source-linked uncertainty.
            continue
        # Case, punctuation and raised-number suffixes have separate source
        # controls. Their disagreement cannot remove an unchanged lexical word.
        if lexical and any(letters(w['text'])==lexical for w in (a,b)):continue
        # An interpolated span box cannot authorize a particular word image.
        # Keep the existing unplaced source control rather than guessing an
        # occurrence inside that span from the alternate OCR output.
        if note.get('mapping')=='unplaced_within_span':continue
        if (len(own_primary)==1 and token.endswith(('-', '\u00ad', '\u2010', '\u2011'))
                and lexical and letters(own_primary[0]['text'])==lexical):continue
        source=note.get('source')or[]
        if (len(source)==1 and source[0].endswith(('-', '\u00ad', '\u2010', '\u2011'))
                and lexical.startswith(letters(source[0]))
                and all(_key(w['text']).strip()==_key(source[0]).strip()for w in (a,b))):
            # This crop owns only the printed prefix. It cannot contradict the
            # complete word whose physical continuation is checked separately.
            continue
        if (any(c.isalpha()for c in token) and any(c.isdigit()for c in token)
                and note.get('source')==note.get('model')==note.get('chosen')==[token]):
            # A letter/number identifier is not a guessed prose spelling. Keep
            # its literal instance and unresolved source control unless an
            # earlier positive adjudication actually proved the replacement.
            continue
        if any(_key(w['text'])==_key(token)for w in (a,b)):continue
        literal=lambda t:''.join(c for c in unicodedata.normalize('NFKD',t.casefold())if c.isalnum())
        if any(SequenceMatcher(None,literal(w['text']),literal(result['tokens'][index])).ratio()<.65 for w in (a,b)):continue
        note=(result.setdefault('word_notes',{}).get(index)or dict(source=[result['tokens'][index]],
            model=[result['tokens'][index]],chosen=[result['tokens'][index]],status='UNRESOLVED'))
        note=dict(note,source_image_only=True,source_bbox=a['bbox'],method='unsettled_alternate_pixel_readings')
        result['word_notes'][index]=note


def below_dot_absent(display,word):
    """Prove a missing below-baseline mark from untouched complete source ink.

    This is restricted to alphabetic words with no descenders. Multiple upright
    letter components must establish one baseline, and the entire lower strip
    must be blank. Cropped bottoms, punctuation, noise and ambiguous baselines
    all decline the proof rather than guessing a diacritic.
    """
    from . import extract,vision_pixels
    import statistics
    literal=''.join(c for c in unicodedata.normalize('NFD',word['text'])if not unicodedata.combining(c))
    if any(c.casefold()in 'gjpqy'for c in literal) or any(not(c.isalpha()or c=='-')for c in literal):return False
    a=word['bbox'];height=a[3]-a[1]
    bounds=extract.pymupdf.Rect(a[0]-.5,a[1]-.1*height,a[2]+.5,a[3]+.5*height)
    if any(bounds[i]<display.rect[i]for i in (0,1))or any(bounds[i]>display.rect[i]for i in (2,3)):return False
    pix=display.pixmap(bounds,scale=8,max_pixels=300_000)
    picture,components=vision_pixels._word_ink_components(pix)
    if not components:return False
    tallest=max(c[3]-c[1]for c in components)
    tall=[c for c in components if c[3]-c[1]>=tallest*.5]
    if len(tall)<3:return False
    baseline=statistics.median(c[3]for c in tall)
    if max(abs(c[3]-baseline)for c in tall)>max(2,tallest*.15):return False
    start=int(baseline+max(1,tallest*.06))
    if picture.height-start<tallest*.25:return False
    return picture.crop((0,start,picture.width,picture.height)).getextrema()[0]>=180


def alternate_regions(regions):
    """Give coherent consecutive rows context without joining other columns.

    These are recognition frames only. Every resulting decision still needs its
    own original word geometry; a rectangle never supplies lexical ownership.
    """
    result=[];previous=None
    for region in regions:
        box=list(region)
        if previous is not None:
            height=max(previous[3]-previous[1],box[3]-box[1])
            overlap=min(previous[2],box[2])-max(previous[0],box[0])
            same_column=overlap>=.8*min(previous[2]-previous[0],box[2]-box[0])
            consecutive=0<=box[1]-previous[3]<=height*1.5
            if same_column and consecutive:
                merged=result[-1];result[-1]=[min(merged[0],box[0]),min(merged[1],box[1]),max(merged[2],box[2]),max(merged[3],box[3])]
            else:result.append(box)
        else:result.append(box)
        previous=box
    return result



def _complete_word_occurrence(source,word):
    """A partial glyph crop cannot stand for either complete word occurrence."""
    if not _same_occurrence(source,word):return False
    a,b=source['bbox'],word['bbox']
    if source.get('native_parts') and not all(
            b[0]-.2<=(p['bbox'][0]+p['bbox'][2])/2<=b[2]+.2 and
            b[1]-.2<=(p['bbox'][1]+p['bbox'][3])/2<=b[3]+.2
            for p in source['native_parts']):return False
    return min(a[2],b[2])-max(a[0],b[0])>=.6*max(a[2]-a[0],b[2]-b[0])


def _complete_recognition(source,word):
    """Locating a word is weaker than certifying all of its printed letters."""
    if not _complete_word_occurrence(source,word):return False
    a,b=source['bbox'],word['bbox']
    return (min(a[2],b[2])-max(a[0],b[0])>=.9*max(a[2]-a[0],b[2]-b[0])
            and min(a[3],b[3])-max(a[1],b[1])>=.55*(a[3]-a[1]))
