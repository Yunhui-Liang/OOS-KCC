"""OOS-KCC core with submission-exact training and bounded-memory fast inference.

The frozen submission protocol remains available via base_protocol='fixed'.
Optional randomized ensemble-generation protocols are for robustness studies only.
"""
from dataclasses import dataclass
from typing import List, Sequence, Optional
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy import sparse

EPS = 1e-12

def one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    labels=np.asarray(labels,dtype=np.int32).reshape(-1)
    z=np.zeros((labels.size,k),dtype=np.float32); z[np.arange(labels.size),labels]=1.0
    return z

def one_hot_sparse(labels: np.ndarray, k: int):
    """CSR equivalent of one_hot, for consensus matrices too large to densify."""
    labels=np.asarray(labels,dtype=np.int32).reshape(-1)
    rows=np.arange(labels.size,dtype=np.int32)
    return sparse.csr_matrix((np.ones(labels.size,dtype=np.float32),(rows,labels)),
                             shape=(labels.size,int(k)),dtype=np.float32)

def compiler(base_labels, consensus_labels, kb, k):
    b=np.asarray(base_labels,dtype=np.int32).reshape(-1); c=np.asarray(consensus_labels,dtype=np.int32).reshape(-1)
    if b.size!=c.size: raise ValueError('base_labels and consensus_labels length mismatch')
    q=np.empty((kb,k),dtype=np.float32)
    prior=np.bincount(c,minlength=k).astype(np.float64); prior/=max(prior.sum(),1.0)
    for j in range(kb):
        m=b==j
        if m.any():
            row=np.bincount(c[m],minlength=k).astype(np.float64); q[j]=row/row.sum()
        else: q[j]=prior
    return q

def top2_membership(distances):
    dmat=np.asarray(distances,dtype=float)
    if dmat.ndim!=2 or dmat.shape[1]<2: raise ValueError('distances must be (n,k>=2)')
    idx=np.argpartition(dmat,1,axis=1)[:,:2]; d=np.take_along_axis(dmat,idx,axis=1)
    order=np.argsort(d,axis=1,kind='stable'); idx=np.take_along_axis(idx,order,axis=1); d=np.take_along_axis(d,order,axis=1)
    d1,d2=d[:,0],d[:,1]; den=d1+d2; exact=d1<=EPS
    w1=np.where(exact,1.0,d2/np.maximum(den,EPS)); w2=np.where(exact,0.0,d1/np.maximum(den,EPS))
    u=np.zeros_like(dmat,dtype=np.float32); rr=np.arange(len(dmat)); u[rr,idx[:,0]]=w1; u[rr,idx[:,1]]=w2
    return u

@dataclass
class BaseRecord:
    view:int; replicate:int; model:object; q:np.ndarray; feature_idx:np.ndarray
    fit_seed:int; n_clusters:int; sample_fraction:float; feature_fraction:float

class OOSKCC:
    def __init__(self,n_clusters:int,base_models_per_view:int=3,base_n_init:int=2,base_max_iter:int=150,
                 kcc_n_init:int=8,kcc_max_iter:int=200,random_state:int=0,standardize:bool=True,
                 base_protocol:str='fixed',cluster_jitter:float=0.0,sample_fraction_min:float=1.0,
                 feature_fraction_min:float=1.0,base_algorithm:str='kmeans'):
        self.n_clusters=int(n_clusters); self.base_models_per_view=int(base_models_per_view)
        self.base_n_init=int(base_n_init); self.base_max_iter=int(base_max_iter)
        self.kcc_n_init=int(kcc_n_init); self.kcc_max_iter=int(kcc_max_iter)
        self.random_state=int(random_state); self.standardize=bool(standardize)
        self.base_protocol=str(base_protocol); self.cluster_jitter=float(cluster_jitter)
        self.sample_fraction_min=float(sample_fraction_min); self.feature_fraction_min=float(feature_fraction_min)
        self.base_algorithm=str(base_algorithm)

    def _check_views(self,views):
        xs=[x.tocsr().astype(np.float32,copy=False) if sparse.issparse(x)
            else np.asarray(x,dtype=np.float32) for x in views]
        if not xs: raise ValueError('at least one view required')
        n=xs[0].shape[0]
        if any(x.ndim!=2 or x.shape[0]!=n for x in xs): raise ValueError('all views must be 2D with same rows')
        return xs

    def _member_spec(self,v,r,n,d):
        # Independent, deterministic seed ledger. Randomized protocols are never used by submission mode.
        if self.base_protocol == 'fixed':
            # Historical submission seed: do not change; exact reproduction depends on it.
            seed=int(self.random_state*1000 + v*100 + r)
        else:
            seed=int((self.random_state+1)*1000003 + (v+1)*1009 + (r+1)*97)
        rng=np.random.default_rng(seed)
        kb=self.n_clusters; sf=1.0; ff=1.0
        if self.base_protocol in ('random_k','diverse') and self.cluster_jitter>0:
            delta=max(1,int(round(self.n_clusters*self.cluster_jitter)))
            lo=max(2,self.n_clusters-delta); hi=max(lo,self.n_clusters+delta)
            kb=int(rng.integers(lo,hi+1))
            kb=min(kb,max(2,n-1))
        if self.base_protocol=='diverse':
            sf=float(rng.uniform(np.clip(self.sample_fraction_min,0.2,1.0),1.0))
            ff=float(rng.uniform(np.clip(self.feature_fraction_min,0.2,1.0),1.0))
        nf=max(1,min(d,int(round(ff*d))))
        feat=np.sort(rng.choice(d,size=nf,replace=False)).astype(np.int32) if nf<d else np.arange(d,dtype=np.int32)
        ns=max(kb+1,min(n,int(round(sf*n))))
        samp=np.sort(rng.choice(n,size=ns,replace=False)).astype(np.int32) if ns<n else np.arange(n,dtype=np.int32)
        return seed,kb,sf,ff,feat,samp

    def fit(self,views):
        xs=self._check_views(views); self.n_views_=len(xs); self.scalers_=[]; xtr=[]
        for x in xs:
            if self.standardize:
                # Sparse inputs cannot be mean-centered without densifying.  Omitting
                # the mean is distance-equivalent for Euclidean K-means: translating
                # every point and centroid by the same per-feature mean leaves all
                # pairwise distances, partitions, and predictions unchanged.
                sc=StandardScaler(with_mean=not sparse.issparse(x)).fit(x)
                z=sc.transform(x).astype(np.float32,copy=False); self.scalers_.append(sc)
            else: sc=None; z=x.copy(); self.scalers_.append(None)
            xtr.append(z)
        # The consensus input is a concatenation of one-hot base partitions.
        # Keep it sparse when its dense form would exceed about 100 MB; sklearn's
        # KMeans accepts CSR and optimizes the same Euclidean objective.
        consensus_cells = int(xtr[0].shape[0]) * self.n_clusters * self.n_views_ * self.base_models_per_view
        sparse_consensus = consensus_cells > 25_000_000
        blocks=[]; temp=[]; ledger=[]; base_labs=[]
        for v,x in enumerate(xtr):
            n,d=x.shape
            for r in range(self.base_models_per_view):
                seed,kb,sf,ff,feat,samp=self._member_spec(v,r,n,d); xf=x[:,feat]
                cls=MiniBatchKMeans if self.base_algorithm=='minibatch' else KMeans
                kwargs=dict(n_clusters=kb,random_state=seed,max_iter=self.base_max_iter)
                if cls is KMeans: kwargs['n_init']=self.base_n_init
                else: kwargs.update(n_init=self.base_n_init,batch_size=min(4096,max(256,len(samp))))
                km=cls(**kwargs).fit(xf[samp])
                lab=km.predict(xf).astype(np.int32)
                blocks.append(one_hot_sparse(lab,kb) if sparse_consensus else one_hot(lab,kb))
                base_labs.append(lab.copy()); temp.append((v,r,km,lab,feat,seed,kb,sf,ff))
                ledger.append({'view':v,'member':r,'seed':seed,'k_base':kb,'sample_fraction':sf,'feature_fraction':ff,
                               'n_features':len(feat),'fit_samples':len(samp),'inertia':float(km.inertia_),
                               'n_iter':int(getattr(km,'n_iter_',-1)),'algorithm':self.base_algorithm})
        htr=sparse.hstack(blocks,format='csr',dtype=np.float32) if sparse_consensus else np.concatenate(blocks,axis=1)
        # preserve historical submission KCC seed exactly for fixed protocol
        kcc_seed=777+self.random_state
        self.kcc_=KMeans(n_clusters=self.n_clusters,n_init=self.kcc_n_init,max_iter=self.kcc_max_iter,random_state=kcc_seed).fit(htr)
        self.consensus_labels_=self.kcc_.labels_.astype(np.int32); self.base_records_=[]
        for v,r,km,lab,feat,seed,kb,sf,ff in temp:
            q=compiler(lab,self.consensus_labels_,kb,self.n_clusters)
            self.base_records_.append(BaseRecord(v,r,km,q,feat,seed,kb,sf,ff))
        self.base_ledger_=ledger; self.base_training_labels_=base_labs; self.training_partition_=htr
        return self

    def _scale_chunk(self,x,v):
        if sparse.issparse(x):
            x=x.tocsr().astype(np.float32,copy=False)
        else:
            x=np.asarray(x,dtype=np.float32)
        sc=self.scalers_[v]
        if sc is None: return x
        if sparse.issparse(x):
            return sc.transform(x).astype(np.float32,copy=False)
        # Explicit float32 path avoids sklearn's extra large temporary arrays during stream inference.
        mean=np.asarray(sc.mean_,dtype=np.float32); scale=np.asarray(sc.scale_,dtype=np.float32)
        return (x-mean)/scale

    @staticmethod
    def _sqdist(x,centers):
        c=np.asarray(centers,dtype=np.float32)
        c2=np.einsum('ij,ij->i',c,c,dtype=np.float32)[None,:]
        if sparse.issparse(x):
            x=x.tocsr().astype(np.float32,copy=False)
            x2=np.asarray(x.multiply(x).sum(axis=1),dtype=np.float32).reshape(-1,1)
            cross=np.asarray(x.dot(c.T),dtype=np.float32)
        else:
            x=np.asarray(x,dtype=np.float32)
            x2=np.einsum('ij,ij->i',x,x,dtype=np.float32)[:,None]
            cross=x@c.T
        d=x2+c2-2.0*cross; np.maximum(d,0,out=d); return d

    def _score_chunk(self,views,mode='soft'):
        n=views[0].shape[0]; total=np.zeros((n,self.n_clusters),dtype=np.float32)
        by_view=[[] for _ in range(self.n_views_)]
        for rec in self.base_records_: by_view[rec.view].append(rec)
        for v,recs in enumerate(by_view):
            x=self._scale_chunk(views[v],v); vs=np.zeros((n,self.n_clusters),dtype=np.float32)
            # Group by feature subset because fixed submission protocol then becomes one GEMM per view.
            groups={}
            for rec in recs: groups.setdefault(tuple(rec.feature_idx.tolist()),[]).append(rec)
            for fkey,grp in groups.items():
                feat=np.asarray(fkey,dtype=np.int32); xx=x[:,feat]
                centers=np.concatenate([np.asarray(r.model.cluster_centers_,dtype=np.float32) for r in grp],axis=0)
                d2=self._sqdist(xx,centers); off=0
                for rec in grp:
                    kb=rec.n_clusters; dm=d2[:,off:off+kb]; off+=kb
                    if mode=='hard':
                        a=np.argmin(dm,axis=1); p=rec.q[a]
                    else:
                        idx=np.argpartition(dm,1,axis=1)[:,:2]; ds=np.take_along_axis(dm,idx,axis=1)
                        order=np.argsort(ds,axis=1,kind='stable'); idx=np.take_along_axis(idx,order,axis=1); ds=np.take_along_axis(ds,order,axis=1)
                        if mode=='soft':
                            d1=np.sqrt(ds[:,0],dtype=np.float32); d2s=np.sqrt(ds[:,1],dtype=np.float32)
                            den=d1+d2s; exact=d1<=EPS; w1=np.where(exact,1.0,d2s/np.maximum(den,EPS)).astype(np.float32); w2=1.0-w1
                        elif mode=='legacy_squared':
                            inv1=1.0/np.maximum(ds[:,0],EPS); inv2=1.0/np.maximum(ds[:,1],EPS); den=inv1+inv2
                            w1=(inv1/den).astype(np.float32); w2=(inv2/den).astype(np.float32)
                        else: raise ValueError("mode must be hard, soft, legacy_squared")
                        rr=np.arange(n); p=w1[:,None]*rec.q[idx[:,0]]+w2[:,None]*rec.q[idx[:,1]]
                    vs+=p.astype(np.float32,copy=False)
            vs/=max(len(recs),1); total+=vs
        total/=self.n_views_; return total

    def predict_scores(self,views,mode='soft',batch_size:Optional[int]=None):
        xs=self._check_views(views)
        if len(xs)!=self.n_views_: raise ValueError('view count mismatch')
        n=xs[0].shape[0]; bs=n if not batch_size else max(1,int(batch_size)); out=np.empty((n,self.n_clusters),dtype=np.float32)
        for s in range(0,n,bs): out[s:s+bs]=self._score_chunk([x[s:s+bs] for x in xs],mode)
        return out

    def predict(self,views,mode='soft',batch_size:Optional[int]=None):
        return self.predict_scores(views,mode,batch_size).argmax(axis=1).astype(np.int32)

    def predict_kcc_hard(self,views,batch_size:Optional[int]=None):
        xs=self._check_views(views); n=xs[0].shape[0]; bs=n if not batch_size else max(1,int(batch_size)); out=np.empty(n,dtype=np.int32)
        for s in range(0,n,bs):
            blocks=[]; chunks=[self._scale_chunk(x[s:s+bs],v) for v,x in enumerate(xs)]
            for rec in self.base_records_:
                lab=rec.model.predict(chunks[rec.view][:,rec.feature_idx]); blocks.append(one_hot(lab,rec.n_clusters))
            out[s:s+bs]=self.kcc_.predict(np.concatenate(blocks,axis=1))
        return out

    def stream_predict_repeated(self,source_views,n_samples,batch_size=32768,mode='soft',jitter=0.0,seed=0):
        """Bounded-memory stream benchmark. Yields label chunks without materializing n_samples."""
        src=self._check_views(source_views); ns=src[0].shape[0]; rng=np.random.default_rng(seed)
        done=0
        while done<n_samples:
            b=min(batch_size,n_samples-done); idx=(np.arange(done,done+b,dtype=np.int64)%ns).astype(np.int64)
            chunk=[]
            for x in src:
                z=x[idx].astype(np.float32,copy=True)
                if jitter>0:
                    if sparse.issparse(z):
                        raise ValueError('Gaussian jitter is not defined for sparse views')
                    z += rng.normal(0,jitter,size=z.shape).astype(np.float32)
                chunk.append(z)
            yield self.predict(chunk,mode=mode,batch_size=b)
            done+=b

    def deployment_model_size_bytes(self):
        total=0
        for rec in self.base_records_: total+=np.asarray(rec.model.cluster_centers_).nbytes+rec.q.nbytes+rec.feature_idx.nbytes
        for sc in self.scalers_:
            if sc is not None: total+=np.asarray(sc.mean_).nbytes+np.asarray(sc.scale_).nbytes
        return int(total)

    def training_artifact_size_bytes(self):
        return int(self.deployment_model_size_bytes()+np.asarray(self.kcc_.cluster_centers_).nbytes)

    def model_size_bytes(self): return self.deployment_model_size_bytes()
