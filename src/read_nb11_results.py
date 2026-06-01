import json, os

base = r'c:\Users\alexy\Documents\Claude_projects\Kaggle competition\ESA_comp\outputs\nb11'
with open(os.path.join(base, 'nb11_results.json')) as f:
    r = json.load(f)

print('=== nb11 L2-SP SWEEP RESULTS ===')
print()
print(f"{'Config':<22} {'Lambda':>8} {'Supp':>7} {'S_gain':>8} {'Pres':>7} {'P_loss':>8} {'PROXY':>8}")
print('-'*75)
for key, v in r.items():
    lam_str = '{:.0e}'.format(v['lambda']) if v['lambda'] is not None else '---'
    print(f"{key:<22} {lam_str:>8} {v['suppression']:>7.4f} {v['supp_gain']:>8.4f} {v['preservation']:>7.4f} {v['pres_loss']:>8.4f} {v['proxy']:>8.4f}")

trained = {k:v for k,v in r.items() if k != 'reference'}
best = max(trained, key=lambda k: trained[k]['proxy'])
print()
print('Best: ' + best + '  proxy=' + str(round(r[best]['proxy'],4)))
print('  suppression_now  = ' + str(round(r[best]['suppression'],4)))
print('  preservation_now = ' + str(round(r[best]['preservation'],4)))
print('  lambda           = ' + str(r[best]['lambda']))
