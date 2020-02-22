f = open('ace-bn-2003/dev', encoding='utf-8')
f_out = open('ace-bn-2003/valid.iob', 'w+', encoding='utf-8')


cur_tag = 'NONE'

for line in f:
    w, tag, _ = line.split()
    if tag.strip() == 'PER':
        tag = 'PERS'

    if tag == 'O':
        f_out.write(' '.join([w,tag]) + '\n')
        cur_tag= 'O'
        continue

    if tag == cur_tag:
        f_out.write(w + ' ' + 'I-' + tag + '\n')
    else:
        cur_tag = tag
        f_out.write(w + ' ' + 'B-' + tag + '\n')
