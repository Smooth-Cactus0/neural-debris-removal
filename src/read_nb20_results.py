import json, os

base = r'c:\Users\alexy\Documents\Claude_projects\Kaggle competition\ESA_comp\outputs\nb20'
with open(os.path.join(base, 'nb20_results.json')) as f:
    r = json.load(f)

print('=== nb20 FINE-PRUNING RESULTS ===')
print()
print(f"{'Config':<15} {'k':>5} {'Supp':>7} {'S_gain':>8} {'Pres':>7} {'P_loss':>8} {'PROXY':>8}")
print('-'*65)
for key, v in r.items():
    print(f"{key:<15} {v['k']:>5} {v['suppression']:>7.4f} {v['supp_gain']:>8.4f} {v['preservation']:>7.4f} {v['pres_loss']:>8.4f} {v['proxy']:>8.4f}")

trained = {k:v for k,v in r.items() if k != 'reference'}
if trained:
    best = max(trained, key=lambda k: trained[k]['proxy'])
    print()
    print('Best: ' + best + '  k=' + str(r[best]['k']) + '  proxy=' + str(round(r[best]['proxy'],4)))
    print('  suppression_now  = ' + str(round(r[best]['suppression'],4)))
    print('  preservation_now = ' + str(round(r[best]['preservation'],4)))
