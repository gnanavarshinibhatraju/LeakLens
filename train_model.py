import pandas as pd, numpy as np, json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
m=pd.read_csv('merged_dataset.csv',parse_dates=['timestamp'])
p=pd.read_csv('/mnt/user-data/uploads/products.csv')
inv=pd.read_csv('/mnt/user-data/uploads/inventory.csv',parse_dates=['date'])
# product-level features (no realised profit used as input)
g=m.groupby('product_id').agg(n_txn=('quantity','size'),units=('quantity','sum'),avg_discount=('discount_pct','mean'),
    avg_price=('unit_price','mean'),share_below_cost=('dq_sold_below_cost','mean')).reset_index()
d=p.merge(g,on='product_id')
d['list_to_cost']=d.list_price/d.unit_cost
d['avg_price_to_cost']=d.avg_price/d.unit_cost
d['gap_per_unit']=d.list_price-d.unit_cost
feats=['list_to_cost','avg_price_to_cost','share_below_cost','avg_discount','gap_per_unit']
X=StandardScaler().fit_transform(d[feats])
iso=IsolationForest(n_estimators=300,contamination=0.34,random_state=42).fit(X)
d['anomaly_score']=-iso.score_samples(X)           # higher = more anomalous
# outcome used ONLY for evaluation: realised profit on completed sales
c=m[m.is_completed==1].groupby('product_id').agg(realised_revenue=('net_revenue','sum'),realised_profit=('gross_profit','sum')).reset_index()
d=d.merge(c,on='product_id',how='left').fillna({'realised_revenue':0,'realised_profit':0})
d['loses_money']=(d.realised_profit<0).astype(int)
y=d.loses_money
k=int(y.sum())
top=d.nlargest(k,'anomaly_score'); rule=d[d.list_price<d.unit_cost]
res={'n_products':len(d),'n_losing':k,
 'iforest_auc':round(roc_auc_score(y,d.anomaly_score),3),'iforest_pr_auc':round(average_precision_score(y,d.anomaly_score),3),
 'iforest_precision_at_k':round(top.loses_money.mean(),3),'iforest_recall_at_k':round(top.loses_money.sum()/k,3),
 'rule_flagged':len(rule),'rule_precision':round(rule.loses_money.mean(),3),'rule_recall':round(rule.loses_money.sum()/k,3),
 'random_precision':round(y.mean(),3)}
print(res)
d['flag']=d.anomaly_score.rank(ascending=False)<=k
d.to_csv('scored_products.csv',index=False)
# stock risk: latest snapshot per product vs recent selling speed
last=inv.sort_values('date').groupby('product_id').tail(1)[['product_id','date','stock_on_hand','reorder_threshold','supplier_delay_days']]
vel=m[~m.is_cancelled.astype(bool)].groupby('product_id').quantity.sum()/365
s=d[['product_id','category','supplier_lead_days']].merge(last,on='product_id'); s['per_day']=s.product_id.map(vel)
s['cover_days']=(s.stock_on_hand/s.per_day).round(1); s['below_reorder']=s.stock_on_hand<s.reorder_threshold
s['lead_total']=s.supplier_lead_days+s.supplier_delay_days
s['at_risk']=s.cover_days<s.lead_total
print('stock at risk',int(s.at_risk.sum()),'below reorder',int(s.below_reorder.sum()),'both',int((s.at_risk&s.below_reorder).sum()))
s.to_csv('stock_risk.csv',index=False)
# dashboard payload
st=m.groupby('status').net_revenue.sum().round(0).to_dict()
cat=m[m.is_completed==1].groupby('category').agg(rev=('net_revenue','sum'),gp=('gross_profit','sum')).round(0)
cat['m']=(cat.gp/cat.rev).round(3)
fl=d[d.flag].sort_values('realised_profit').head(40)
prods=[dict(id=r.product_id,cat=r.category,cost=r.unit_cost,list=r.list_price,disc=round(r.avg_discount*100,1),
  below=round(r.share_below_cost*100),score=round(r.anomaly_score,3),rev=round(r.realised_revenue),gp=round(r.realised_profit),units=int(r.units)) for r in fl.itertuples()]
stk=s[s.at_risk&s.below_reorder].sort_values('cover_days').head(10)
stock=[dict(id=r.product_id,cat=r.category,stock=int(r.stock_on_hand),thr=int(r.reorder_threshold),cover=r.cover_days,lead=int(r.lead_total)) for r in stk.itertuples()]
json.dump(dict(res=res,status=st,cat=cat.reset_index().to_dict('records'),prods=prods,stock=stock,
  tot=dict(rev=round(m.net_revenue.sum()),comp_gp=round(c.realised_profit.sum()),loss=round(d[d.loses_money==1].realised_profit.sum()),nstockrisk=int((s.at_risk&s.below_reorder).sum()))),open('/home/claude/data.json','w'))
