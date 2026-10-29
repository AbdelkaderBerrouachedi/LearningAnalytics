"""Canonical, auditable OULAD experiment. No filesystem mutations at import time."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, logging, platform, sys, time, urllib.request, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
PROJECT = CODE.parent
KEY = ['code_module', 'code_presentation', 'id_student']
CUTOFFS = {2:14,4:28,6:42,8:56}
SEED = 42
DEMOGRAPHICS = ['gender_binary','disability_binary','age_band_ord','education_ord','imd_band_ord']
BACKGROUND = ['studied_credits','num_of_prev_attempts','date_registration']
ACTIVITIES = ['forumng','oucontent','resource','subpage','url','quiz','homepage','other']
ENGAGEMENT = ['total_clicks','active_days','distinct_sites','clicks_per_active_day','pre_course_clicks','pre_course_active_days',
              'clicks_last_7d','active_days_last_7d','clicks_last_14d','active_days_last_14d','days_since_last_click',
              'click_ratio_7d_to_14d','active_day_ratio_7d_to_14d','recent_to_total_click_share'] + ['clicks_'+a for a in ACTIVITIES]
ASSESSMENT = ['assessments_due','assessments_submitted','missed_assessments','submission_rate','mean_submission_delay','late_submissions_count']
FEATURES = BACKGROUND + DEMOGRAPHICS + ENGAGEMENT + ASSESSMENT
SUBSETS = {'full':FEATURES,'without_demographics':BACKGROUND+ENGAGEMENT+ASSESSMENT,
           'background_only':BACKGROUND+DEMOGRAPHICS,'vle_only':ENGAGEMENT,'assessment_only':ASSESSMENT,
           'with_scores_assumption':FEATURES+['mean_assessment_score']}
RAW_FILES = ['courses','studentInfo','studentRegistration','vle','assessments','studentAssessment','studentVle']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def download_data(target):
    target=Path(target); target.mkdir(parents=True,exist_ok=True)
    if all((target/(f+'.csv')).exists() for f in RAW_FILES):return
    urls=['https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip',
          'https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip']
    errors=[]
    for url in urls:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'OULAD-reproducibility/2.0'})
            archive=target/'download.zip'
            with urllib.request.urlopen(req,timeout=90) as response, archive.open('wb') as f:
                while block:=response.read(2**20):f.write(block)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    basename=Path(name).name
                    if basename in [f+'.csv' for f in RAW_FILES]:
                        (target/basename).write_bytes(z.read(name))
                # Some UCI releases wrap OULAD.zip inside the outer archive.
                for name in z.namelist():
                    if name.lower().endswith('.zip'):
                        import io
                        with zipfile.ZipFile(io.BytesIO(z.read(name))) as inner:
                            for member in inner.namelist():
                                basename=Path(member).name
                                if basename in [f+'.csv' for f in RAW_FILES]: (target/basename).write_bytes(inner.read(member))
            if not all((target/(f+'.csv')).exists() for f in RAW_FILES): raise ValueError('Incomplete UCI archive')
            write_json(target/'provenance.json',{'url':url,'archive_sha256':sha(archive),'files':{f:sha(target/(f+'.csv')) for f in RAW_FILES}})
            return
        except Exception as e:errors.append(str(e))
    raise RuntimeError('OULAD download failed: '+'; '.join(errors))

def load_data(raw):
    raw=Path(raw)
    missing=[f for f in RAW_FILES if not (raw/(f+'.csv')).exists()]
    if missing:raise FileNotFoundError('Missing real data: '+', '.join(missing)+'. Use --download explicitly.')
    d={f:pd.read_csv(raw/(f+'.csv'),na_values=['?','']) for f in RAW_FILES}
    for table,keys in [('studentInfo',KEY),('studentRegistration',KEY),('courses',KEY[:2]),('assessments',['id_assessment'])]:
        if d[table].duplicated(keys).any():raise ValueError('Duplicate key in '+table)
    base=d['studentInfo'].merge(d['studentRegistration'],on=KEY,validate='one_to_one').merge(d['courses'],on=KEY[:2],validate='many_to_one')
    sites=d['vle'][['code_module','code_presentation','id_site','activity_type']]
    if sites.duplicated(['code_module','code_presentation','id_site']).any():raise ValueError('Duplicate site key')
    d['studentVle']=d['studentVle'].merge(sites,on=['code_module','code_presentation','id_site'],how='left',validate='many_to_one')
    if d['studentVle'].activity_type.isna().any():raise ValueError('Unmapped VLE site')
    known={'0-10%','10-20','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%'}
    if not set(base.imd_band.dropna()).issubset(known):raise ValueError('Unknown IMD category')
    if not set(base.gender.dropna()).issubset({'F','M'}):raise ValueError('Unknown gender category')
    d['base']=base
    return d

def snapshot(d,t,include_unknown_registration=False):
    """At end of day t: registration <=t; event in (t, course end]."""
    base=d['base']
    ambiguous=(base.final_result.eq('Withdrawn') & base.date_unregistration.isna())
    outside=base.date_unregistration.gt(base.module_presentation_length)
    mismatch=base.final_result.ne('Withdrawn') & base.date_unregistration.notna()
    eligible=(~ambiguous & ~outside & ~mismatch & (base.date_unregistration.isna()|base.date_unregistration.gt(t)))
    registered=base.date_registration.le(t)
    if include_unknown_registration: registered |= base.date_registration.isna()
    f=base.loc[eligible & registered].copy()
    f['y_withdrawal']=(f.date_unregistration.gt(t)&f.date_unregistration.le(f.module_presentation_length)).astype(int)
    sv=d['studentVle'].loc[d['studentVle'].date.le(t)].copy()
    sv['cat']=sv.activity_type.where(sv.activity_type.isin(ACTIVITIES[:-1]),'other')
    def aggregate(frame,definitions):
        return frame.groupby(KEY).agg(**definitions).reset_index()
    all_agg=aggregate(sv,{'total_clicks':('sum_click','sum'),'active_days':('date','nunique'),'distinct_sites':('id_site','nunique'),'last_day':('date','max')})
    f=f.merge(all_agg,on=KEY,how='left',validate='one_to_one')
    for n in [7,14]:
        sub=sv.loc[sv.date.ge(t-n+1)]
        f=f.merge(aggregate(sub,{f'clicks_last_{n}d':('sum_click','sum'),f'active_days_last_{n}d':('date','nunique')}),on=KEY,how='left',validate='one_to_one')
    f=f.merge(aggregate(sv.loc[sv.date.lt(0)],{'pre_course_clicks':('sum_click','sum'),'pre_course_active_days':('date','nunique')}),on=KEY,how='left',validate='one_to_one')
    pivot=sv.pivot_table(index=KEY,columns='cat',values='sum_click',aggfunc='sum',fill_value=0).reindex(columns=ACTIVITIES,fill_value=0)
    pivot.columns=['clicks_'+str(c) for c in pivot.columns]
    f=f.merge(pivot.reset_index(),on=KEY,how='left',validate='one_to_one')
    for c in ['total_clicks','active_days','distinct_sites','pre_course_clicks','pre_course_active_days','clicks_last_7d','active_days_last_7d','clicks_last_14d','active_days_last_14d']+['clicks_'+a for a in ACTIVITIES]:f[c]=f[c].fillna(0)
    f['days_since_last_click']=(t-f.last_day).fillna(t+30)
    f['clicks_per_active_day']=f.total_clicks/f.active_days.replace(0,np.nan)
    f['clicks_per_active_day']=f.clicks_per_active_day.fillna(0)
    f['click_ratio_7d_to_14d']=f.clicks_last_7d/(f.clicks_last_14d-f.clicks_last_7d+1)
    f['active_day_ratio_7d_to_14d']=f.active_days_last_7d/(f.active_days_last_14d-f.active_days_last_7d+1)
    f['recent_to_total_click_share']=f.clicks_last_14d/(f.total_clicks+1)
    due=d['assessments'].loc[d['assessments'].date.le(t)&d['assessments'].assessment_type.ne('Exam')].copy()
    counts=due.groupby(KEY[:2]).size().rename('assessments_due').reset_index()
    f=f.merge(counts,on=KEY[:2],how='left',validate='many_to_one')
    sa=d['studentAssessment']
    sa=sa.loc[sa.date_submitted.le(t)&sa.is_banked.eq(0)].merge(due,on='id_assessment',validate='many_to_one')
    sa['delay']=sa.date_submitted-sa.date
    sa['late']=(sa.delay>0).astype(int)
    agg=aggregate(sa,{'assessments_submitted':('id_assessment','nunique'),'mean_submission_delay':('delay','mean'),
                      'late_submissions_count':('late','sum'),'mean_assessment_score':('score','mean')})
    f=f.merge(agg,on=KEY,how='left',validate='one_to_one')
    for c in ['assessments_due','assessments_submitted','mean_submission_delay','late_submissions_count']:f[c]=f[c].fillna(0)
    f['missed_assessments']=(f.assessments_due-f.assessments_submitted).clip(lower=0)
    f['submission_rate']=np.where(f.assessments_due>0,f.assessments_submitted/f.assessments_due.replace(0,np.nan),1)
    f['gender_binary']=f.gender.map({'F':1,'M':0})
    f['disability_binary']=f.disability.map({'Y':1,'N':0})
    f['age_band_ord']=f.age_band.map({'0-35':1,'35-55':2,'55<=':3})
    f['education_ord']=f.highest_education.map({'No Formal quals':0,'Lower Than A Level':1,'A Level or Equivalent':2,'HE Qualification':3,'Post Graduate Qualification':4})
    imd={'0-10%':0,'10-20':1,'10-20%':1,**{f'{n}-{n+10}%':n//10 for n in range(20,100,10)}}
    f['imd_band_ord']=f.imd_band.map(imd)
    if f.duplicated(KEY).any():raise AssertionError('Duplicate enrollment after extraction')
    assert (f.date_registration.le(t)| (include_unknown_registration & f.date_registration.isna())).all()
    return f.reset_index(drop=True)

def budget_weights(scores,budget):
    """Expected exact top-k allocation, uniform random choice inside a tied block."""
    scores=np.asarray(scores,float)
    if not len(scores):return np.array([])
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    k=int(np.floor(budget*len(scores)))
    if k<=0:return np.zeros(len(scores))
    if k>=len(scores):return np.ones(len(scores))
    boundary=np.partition(scores,len(scores)-k)[len(scores)-k]
    weights=(scores>boundary).astype(float)
    tied=scores==boundary
    weights[tied]=(k-weights.sum())/tied.sum()
    assert np.isclose(weights.sum(),k)
    return weights

def metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    two=len(np.unique(y))==2
    out={'ap':float(average_precision_score(y,p)) if y.sum() else None,
         'auroc':float(roc_auc_score(y,p)) if two else None,'brier':float(brier_score_loss(y,p)),'prevalence':float(y.mean()),'n':len(y)}
    for b in [.05,.1,.2]:
        w=budget_weights(p,b);hits=float(w@y);k=w.sum();key=str(int(b*100))
        out['precision_'+key]=hits/k if k else None
        out['recall_'+key]=hits/y.sum() if y.sum() else None
    return out

def bootstrap(y,p,ids,n=1000,other=None):
    """Student-cluster percentile CIs and paired differences, fixed trained model."""
    y=np.asarray(y);p=np.asarray(p);ids=np.asarray(ids)
    groups={g:np.flatnonzero(ids==g) for g in np.unique(ids)};unique=list(groups)
    rng=np.random.default_rng(SEED);values={k:[] for k in ['ap','auroc','brier','precision_5','recall_5','precision_10','recall_10','precision_20','recall_20']}
    for _ in range(n):
        idx=np.concatenate([groups[unique[i]] for i in rng.integers(0,len(unique),len(unique))])
        if len(np.unique(y[idx]))<2:continue
        m=metrics(y[idx],p[idx]); ref=metrics(y[idx],np.asarray(other)[idx]) if other is not None else None
        for k in values:
            if m[k] is not None and (ref is None or ref[k] is not None):values[k].append(m[k]-(ref[k] if ref else 0))
    result={}
    for k,v in values.items():
        if v:
            result[k+'_low'],result[k+'_high']=map(float,np.quantile(v,[.025,.975]))
            if other is not None: result[k+'_bootstrap_p']=float(min(1,2*min((np.sum(np.asarray(v)<=0)+1)/(len(v)+1),(np.sum(np.asarray(v)>=0)+1)/(len(v)+1))))
    return result

def model_set(seed=SEED):
    import lightgbm as lgb
    import xgboost as xgb
    return {
      'Logistic':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=seed)),
      'HistGB':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingClassifier(max_iter=150,learning_rate=.03,max_leaf_nodes=31,random_state=seed)),
      'LightGBM':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),lgb.LGBMClassifier(n_estimators=150,learning_rate=.03,num_leaves=31,min_child_samples=30,colsample_bytree=.8,random_state=seed,verbosity=-1,n_jobs=4)),
      'XGBoost':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),xgb.XGBClassifier(n_estimators=150,learning_rate=.03,max_depth=4,subsample=.8,colsample_bytree=.8,random_state=seed,n_jobs=4,eval_metric='logloss'))}

class SigmoidCalibration:
    """Prespecified calibration on independent historical student groups."""
    def fit(self,p,y):
        self.model=LogisticRegression(C=1,max_iter=1000,random_state=SEED).fit(self.logit(p),y)
        return self
    @staticmethod
    def logit(p):
        p=np.clip(np.asarray(p),1e-7,1-1e-7);return np.log(p/(1-p)).reshape(-1,1)
    def predict(self,p):return self.model.predict_proba(self.logit(p))[:,1]

def fit_predict(model,train,cal,test,features):
    start=time.perf_counter();model.fit(train[features],train.y_withdrawal)
    calraw=model.predict_proba(cal[features])[:,1]
    calibration=SigmoidCalibration().fit(calraw,cal.y_withdrawal)
    t0=time.perf_counter();raw=model.predict_proba(test[features])[:,1];latency=time.perf_counter()-t0
    return raw,calibration.predict(raw),model,calibration,time.perf_counter()-start,latency

def explanation(model,train,test,features,week,out):
    import shap
    rng=np.random.default_rng(SEED)
    xt=model.steps[0][1].transform(train[features]);xe=model.steps[0][1].transform(test[features])
    ti=rng.choice(len(xt),min(500,len(xt)),replace=False);ei=rng.choice(len(xe),min(500,len(xe)),replace=False)
    ex=shap.TreeExplainer(model.steps[-1][1],feature_perturbation='tree_path_dependent',model_output='raw')
    trainsh=np.asarray(ex.shap_values(xt[ti]));testsh=np.asarray(ex.shap_values(xe[ei]))
    if trainsh.ndim!=2 or testsh.ndim!=2:raise ValueError('Unexpected SHAP shape')
    ma=np.abs(trainsh).mean(0);mb=np.abs(testsh).mean(0)
    result=pd.DataFrame({'week':week,'feature':features,'train_mean_abs':ma,'test_mean_abs':mb})
    result.to_csv(out/f'shap_week_{week}.csv',index=False)
    np.savez_compressed(out/f'shap_week_{week}.npz',values=testsh,expected_value=np.asarray(ex.expected_value),features=np.asarray(features),**{k:test.iloc[ei][k].to_numpy(dtype=np.int64 if k=='id_student' else str) for k in KEY})
    groups=[]
    for module in sorted(test.code_module.unique()):
        ix=np.flatnonzero(test.code_module.to_numpy()==module);ix=rng.choice(ix,min(len(ix),300),replace=False)
        val=np.asarray(ex.shap_values(xe[ix]));mean=np.abs(val).mean(0)
        for f,v in zip(features,mean):groups.append({'week':week,'module':module,'feature':f,'mean_abs':float(v)})
    pd.DataFrame(groups).to_csv(out/f'shap_modules_week_{week}.csv',index=False)
    rho=float(spearmanr(ma,mb).statistic)
    return {'week':week,'train_test_spearman':rho,'n_train_explained':len(ti),'n_test_explained':len(ei),'top_feature':features[int(np.argmax(mb))],'scale':'uncalibrated model log-odds'}

def fairness(test,p,variant,week):
    weights=budget_weights(p,.1);y=test.y_withdrawal.to_numpy();rows=[]
    for column in ['gender','disability','age_band','imd_band']:
        for group,df in test.groupby(column,dropna=False):
            idx=test.index.get_indexer(df.index);yy=y[idx];ww=weights[idx];pos=yy.sum();neg=len(yy)-pos
            rows.append({'week':week,'variant':variant,'attribute':column,'group':str(group),'n':len(yy),'prevalence':float(yy.mean()),
              'selection_rate':float(ww.mean()),'expected_alerts':float(ww.sum()),'precision':float(ww@yy/ww.sum()) if ww.sum() else None,
              'tpr':float(ww@yy/pos) if pos else None,'fpr':float(ww@(1-yy)/neg) if neg else None})
    return rows

def run(raw,out,mode='real',n_boot=1000,explain=True):
    out=Path(out)
    if mode not in ['real','demo']:raise ValueError('Data mode is mandatory')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Run output must be new/empty: '+str(out))
    if mode=='demo' and 'demo' not in [p.lower() for p in out.parts]:raise ValueError('Demo output requires a demo path component')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'execution.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    start=time.perf_counter();d=load_data(raw);base=d['base']
    if mode=='real' and (len(base)!=32593 or base.id_student.nunique()!=28785):raise ValueError('Real OULAD cardinality validation failed')
    # Historical labels from all 2013 offerings have matured before October 2014.
    # Latest possible 2013J start is 31 October; all OULAD lengths must be < 335 days.
    historical=base.code_presentation.isin(['2013B','2013J'])
    if base.loc[historical,'module_presentation_length'].max()>=335:raise ValueError('Historical label maturity cannot be certified')
    hist_ids=np.sort(base.loc[historical,'id_student'].unique())
    fitix,calix=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(hist_ids,groups=hist_ids))
    fitids=set(hist_ids[fitix]);calids=set(hist_ids[calix]);seen=set(base.loc[base.code_presentation.ne('2014J'),'id_student'])
    assert not fitids&calids
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','shap','matplotlib','joblib']}
    manifest={'data_mode':mode,'status':'running','n_bootstrap':n_boot,'seed':SEED,'primary_model':'Logistic','ranking':'raw probability','calibration':'historical student-disjoint sigmoid',
              'history':['2013B','2013J'],'test':'2014J','test_exclusion':'any id present in a non-2014J enrollment',
              'maturity':'2013 labels available before October 2014; no 2014B outcome used',
              'versions':versions,'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
              'data_sha256':{f:sha(Path(raw)/(f+'.csv')) for f in RAW_FILES},'code_sha256':{str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py') if '__pycache__' not in p.parts},
              'model_parameters':{k:v.get_params(deep=True).__repr__() for k,v in model_set().items()}}
    write_json(out/'manifest.json',manifest)
    write_json(out/'identity_partition.json',{'fit_ids':sorted(map(int,fitids)),'calibration_ids':sorted(map(int,calids))})
    performances=[];cohorts=[];transfers=[];ablations=[];fair=[];deltas=[];stabilities=[];leadtimes=[];unknown=[];recurring=[]
    for week,t in CUTOFFS.items():
        logging.info('Starting week %s',week);f=snapshot(d,t)
        train=f.loc[f.code_presentation.isin(['2013B','2013J'])&f.id_student.isin(fitids)].copy()
        cal=f.loc[f.code_presentation.isin(['2013B','2013J'])&f.id_student.isin(calids)].copy()
        test=f.loc[f.code_presentation.eq('2014J')&~f.id_student.isin(seen)].copy().reset_index(drop=True)
        if min(len(train),len(cal),len(test))==0:raise ValueError('Empty partition')
        assert not set(train.id_student)&set(cal.id_student)
        assert not set(test.id_student)&(fitids|calids)
        for name,df in [('fit',train),('calibration',cal),('test',test)]:
            cohorts.append({'week':week,'partition':name,'enrollments':len(df),'students':df.id_student.nunique(),'withdrawals':int(df.y_withdrawal.sum()),'prevalence':float(df.y_withdrawal.mean())})
            df[KEY+['y_withdrawal']].to_csv(out/f'split_{week}_{name}.csv',index=False)
        cohorts.append({'week':week,'partition':'all_eligible','enrollments':len(f),'students':f.id_student.nunique(),'withdrawals':int(f.y_withdrawal.sum()),'prevalence':float(f.y_withdrawal.mean())})
        y=test.y_withdrawal.to_numpy();ids=test.id_student.to_numpy()
        predictions=test[KEY+['y_withdrawal']].copy()
        predictions['withdrawal_day']=test.date_unregistration; predictions['cutoff_day']=t
        models=model_set();all_scores={};fitted={}
        prior=np.full(len(test),train.y_withdrawal.mean())
        score=.1*test.days_since_last_click.to_numpy()-np.log1p(test.clicks_last_14d.to_numpy())
        rulescore=1/(1+np.exp(-np.clip(score,-30,30)))
        for name,p in [('Prevalence',prior),('ActivityRule',rulescore)]:all_scores[name]=(p,p,0.,0.)
        for name,model in models.items():
            rawp,calp,fm,calibrator,seconds,latency=fit_predict(model,train,cal,test,FEATURES)
            all_scores[name]=(rawp,calp,seconds,latency);fitted[name]=(fm,calibrator)
            joblib.dump({'model':fm,'calibrator':calibrator,'features':FEATURES},out/f'model_{week}_{name}.joblib')
        for name,(p,cp,seconds,latency) in all_scores.items():
            logging.info('Week %s model %s cluster bootstrap (%s)',week,name,n_boot)
            m=metrics(y,p);ci=bootstrap(y,p,ids,n_boot)
            performances.append({'week':week,'model':name,**m,**ci,'calibrated_brier':float(brier_score_loss(y,cp)),
                                 'fit_and_calibration_seconds':seconds,'predict_seconds':latency})
            predictions[name+'_raw']=p;predictions[name+'_calibrated']=cp
        predictions.to_csv(out/f'predictions_week_{week}.csv',index=False)
        primary=all_scores['Logistic'][0]
        for name in ['HistGB','LightGBM','XGBoost','ActivityRule']:
            m=metrics(y,all_scores[name][0]);r=metrics(y,primary)
            deltas.append({'week':week,'comparison':name+' minus Logistic',**{k:m[k]-r[k] for k in ['ap','auroc','brier','precision_10','recall_10']},**bootstrap(y,all_scores[name][0],ids,n_boot,other=primary)})
        for module,df in test.groupby('code_module'):
            idx=test.index.get_indexer(df.index)
            for name in ['Logistic','LightGBM']:
                transfers.append({'week':week,'module':module,'model':name,**metrics(y[idx],all_scores[name][0][idx])})
        for variant,features in SUBSETS.items():
            if variant=='full':p,cp=all_scores['LightGBM'][:2]
            else:
                p,cp,*_=fit_predict(model_set()['LightGBM'],train,cal,test,features)
            predictions[variant+'_lgb_raw']=p
            ablations.append({'week':week,'variant':variant,'features':len(features),**metrics(y,p),'calibrated_brier':float(brier_score_loss(y,cp))})
            if variant in ['full','without_demographics']:
                fair.extend(fairness(test,p,variant,week))
            if variant=='without_demographics':
                ref=all_scores['LightGBM'][0]
                deltas.append({'week':week,'comparison':'without_demographics minus full LightGBM',**{k:metrics(y,p)[k]-metrics(y,ref)[k] for k in ['ap','auroc','brier','precision_10','recall_10']},**bootstrap(y,p,ids,n_boot,other=ref)})
        predictions.to_csv(out/f'predictions_week_{week}.csv',index=False)
        w=budget_weights(primary,.1);pos=(y==1)&(w>0)
        for i in np.flatnonzero(pos):leadtimes.append({'week':week,'id_student':int(ids[i]),'module':test.iloc[i].code_module,'days_before_withdrawal':float(test.iloc[i].date_unregistration-t),'alert_weight':float(w[i])})
        # Sensitivities use the fixed fitted model, with changed populations explicitly labelled.
        for label,pop in [('all_2014J',f.loc[f.code_presentation.eq('2014J')].copy()),
                          ('unknown_registration_included',snapshot(d,t,True).loc[lambda x:x.code_presentation.eq('2014J')&~x.id_student.isin(seen)].copy())]:
            p=fitted['Logistic'][0].predict_proba(pop[FEATURES])[:,1]
            unknown.append({'week':week,'population':label,**metrics(pop.y_withdrawal,p)})
        if explain:stabilities.append(explanation(fitted['LightGBM'][0],train,test,FEATURES,week,out))
        # Incremental exports preserve evidence if a later step fails.
        for filename,rows in [('performance',performances),('cohorts',cohorts),('module_transfer',transfers),('ablations',ablations),('fairness',fair),('paired_differences',deltas),('lead_times',leadtimes),('population_sensitivity',unknown),('explanation_stability',stabilities)]:
            pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
    audit={'raw_enrollments':len(base),'unique_students':base.id_student.nunique(),'missing_registration':int(base.date_registration.isna().sum()),
           'withdrawn_missing_date':int((base.final_result.eq('Withdrawn')&base.date_unregistration.isna()).sum()),
           'withdrawal_after_end':int(base.date_unregistration.gt(base.module_presentation_length).sum()),
           'nonwithdrawn_with_date':int((base.final_result.ne('Withdrawn')&base.date_unregistration.notna()).sum()),
           'late_registration_by_cutoff':{t:int(base.date_registration.gt(t).sum()) for t in CUTOFFS.values()}}
    write_json(out/'data_audit.json',audit)
    figures(out)
    manifest.update(status='completed',elapsed_seconds=time.perf_counter()-start)
    logging.info('Completed in %.1f seconds',manifest['elapsed_seconds'])
    for handler in logging.getLogger().handlers: handler.flush()
    manifest['outputs']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',manifest)
    return out

def figures(run_dir):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    run_dir=Path(run_dir);dest=run_dir/'figures';dest.mkdir(exist_ok=True)
    p=pd.read_csv(run_dir/'performance.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,g in p.groupby('model'):
        if name=='Prevalence':continue
        axes[0].plot(g.week,g.ap,'o-',label=name);axes[1].plot(g.week,g.recall_10,'o-',label=name)
    base=p.loc[p.model.eq('Prevalence')];axes[0].plot(base.week,base.ap,'k--',label='Test prevalence');axes[1].axhline(.1,c='k',ls='--')
    for ax,title,y in zip(axes,['Discrimination','Expected top-10% recall'],['Average precision','Recall']):ax.set(xlabel='Prediction week',ylabel=y,title=title,xticks=[2,4,6,8]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(dest/'performance.pdf');fig.savefig(dest/'performance.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,week in zip(axes,[4,8]):
        df=pd.read_csv(run_dir/f'predictions_week_{week}.csv');ax.plot([0,1],[0,1],'k--')
        for name in ['Logistic_raw','Logistic_calibrated','LightGBM_raw','LightGBM_calibrated']:
            obs,mean=calibration_curve(df.y_withdrawal,df[name],n_bins=6,strategy='quantile');ax.plot(mean,obs,'o-',label=name)
        ax.set(xlabel='Mean predicted risk',ylabel='Observed fraction',title=f'Week {week}');ax.legend(fontsize=6)
    fig.savefig(dest/'calibration.pdf');fig.savefig(dest/'calibration.png',dpi=200);plt.close(fig)
    if (run_dir/'shap_week_4.csv').exists():
        s=pd.read_csv(run_dir/'shap_week_4.csv').sort_values('test_mean_abs').tail(12)
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained');s.plot.barh(x='feature',y=['train_mean_abs','test_mean_abs'],ax=ax)
        ax.set(xlabel='Mean absolute SHAP (raw log-odds)',ylabel='',title='Week 4 LightGBM: same fitted model');fig.savefig(dest/'shap.pdf');fig.savefig(dest/'shap.png',dpi=200);plt.close(fig)

def cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['real','demo'],required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--skip-shap',action='store_true')
    args=parser.parse_args(argv)
    if args.download:
        if args.mode!='real':parser.error('--download requires --mode real')
        download_data(args.raw)
    if args.out is None:
        if args.download:return
        parser.error('--out is required for an experiment')
    if args.bootstrap<1:parser.error('--bootstrap must be positive')
    run(args.raw,args.out,args.mode,args.bootstrap,not args.skip_shap)

if __name__=='__main__':cli()

