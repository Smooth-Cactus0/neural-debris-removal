import json, glob, os

base = r'c:\Users\alexy\Documents\Claude_projects\Kaggle competition\ESA_comp\outputs\nb10'
path = os.path.join(base, 'nb10_results.json')
with open(path) as f:
    r = json.load(f)

print('=== nb10 FREEZE FINETUNE RESULTS ===')
print()
print(f"{'Mode':<35} {'Supp':>7} {'S_gain':>8} {'Pres':>7} {'P_loss':>8} {'PROXY':>8}")
print('-'*80)
for key, v in r.items():
    mode = v['freeze_mode']
    print(f"{mode:<35} {v['suppression']:>7.4f} {v['supp_gain']:>8.4f} {v['preservation']:>7.4f} {v['pres_loss']:>8.4f} {v['proxy']:>8.4f}")

modes = {k:v for k,v in r.items() if k != 'reference'}
best = max(modes, key=lambda k: modes[k]['proxy'])
print()
print('Best mode: ' + best + '  proxy=' + str(round(r[best]['proxy'], 4)))
print('suppression_now = ' + str(round(r[best]['suppression'], 4)))
print('preservation_now = ' + str(round(r[best]['preservation'], 4)))
