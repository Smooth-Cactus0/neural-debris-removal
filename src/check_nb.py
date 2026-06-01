import json, sys
for nb in ['nb10/nb10_freeze_finetune.ipynb']:
    path = 'c:/Users/alexy/Documents/Claude_projects/Kaggle competition/ESA_comp/notebooks/' + nb
    with open(path) as f:
        data = json.load(f)
    cells = data['cells']
    print(nb + ': ' + str(len(cells)) + ' cells, nbformat ' + str(data['nbformat']))
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            cid = c.get('id','?')
            print('  cell ' + str(i) + ' (' + cid + '): ' + str(len(src)) + ' chars')
print('JSON valid')
