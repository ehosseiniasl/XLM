import os

def is_title(line, prev):
  return line[:3] == ' = ' and line[3] != '=' and line[-3] == '=' and len(prev.strip()) == 0

title = 0
prev_line = ''
with open('data/wikitext-103/wiki.train.tokens') as infile:
  article = ''
  in_article = False
  for line in infile:
    if is_title(line, prev_line):
      in_article = True
      if in_article:
        with open(os.path.join('data/wikitext-103/train/', str(title) + '.txt'), 'w') as outfile:
          outfile.write(article)
        article = ''
      title += 1
      article += line
    else:
      article += line
    prev_line = line
