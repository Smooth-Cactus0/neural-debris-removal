import json
path = 'c:/Users/alexy/Documents/Claude_projects/Kaggle competition/ESA_comp/notebooks/nb11/nb11_l2sp_finetune.ipynb'
with open(path) as f:
    data = json.load(f)
cells = data['cells']
print('nb11: ' + str(len(cells)) + ' cells, nbformat ' + str(data['nbformat']))
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        cid = c.get('id','?')
        print('  cell ' + str(i) + ' (' + cid + '): ' + str(len(src)) + ' chars')
print('JSON valid')
