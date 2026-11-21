/* ResidueHead: causal online associative transition memory. No imports, no network.
   Every stored bridge is (address at t, observed delta t->t+1).
   CommonJS export for Node tests; window.ResidueHead for offline HTML. */
(function (root, factory) {
  const Head = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = Head;
  if (root) root.ResidueHead = Head;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (v,lo,hi) => Math.min(hi,Math.max(lo,v));
  const mse = (a,b) => { let s=0; for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;} return s/a.length; };
  class ResidueHead {
    constructor({width=48,height=36,gridX=12,gridY=9,capacity=640,neighbors=6,
      halfLife=420,temperature=0.008,familiarityScale=0.02,feedback=0.06}={}) {
      if (width%gridX || height%gridY) throw Error('Feature grid must divide frame dimensions');
      this.width=width; this.height=height; this.N=width*height;
      this.gridX=gridX; this.gridY=gridY; this.G=gridX*gridY;
      this.capacity=capacity; this.neighbors=neighbors;
      this.halfLife=halfLife; this.temperature=temperature; this.familiarityScale=familiarityScale;
      this.feedback=feedback; this.mode='temporal'; this.learning=true;
      this.reset();
    }
    reset(){this.bank=[];this.t=0;this.previous=null;this.previousAddress=null;
      this.pending=null;this.trials=[];this.latest=null;this.totalModel=0;this.totalHold=0;
      this.totalTrials=0;this.epoch=0;}
    /** Spatial coarse average: one low-frequency address, *not* image reconstruction. */
    gist(frame){
      const result=new Float32Array(this.G),bx=this.width/this.gridX,by=this.height/this.gridY;
      let k=0;
      for(let gy=0;gy<this.gridY;gy++)for(let gx=0;gx<this.gridX;gx++){
        let s=0;
        for(let yy=0;yy<by;yy++)for(let xx=0;xx<bx;xx++)
          s+=frame[(gy*by+yy)*this.width+gx*bx+xx];
        result[k++]=s/(bx*by);
      }
      return result;
    }
    address(frame,previous){
      const coarse=this.gist(frame),motion=new Float32Array(this.G);
      if(previous){const old=this.gist(previous);for(let i=0;i<this.G;i++)motion[i]=coarse[i]-old[i];}
      return {coarse,motion};
    }
    distance(a,b){
      let s=0;
      const weight=this.mode==='temporal'?16:0;
      for(let i=0;i<this.G;i++){
        const d=a.coarse[i]-b.coarse[i];
        const v=a.motion[i]-b.motion[i];
        s+=d*d+weight*v*v;
      }
      return s/this.G;
    }
    /** Only reads completed past transitions. Return predicted *delta* and its memory witnesses. */
    retrieve(address){
      const result=new Float32Array(this.N);
      if(!this.bank.length) return {delta:result,confidence:0,distance:null,hits:[]};
      const best=[];
      for(let i=0;i<this.bank.length;i++){
        const d=this.distance(address,this.bank[i].key);
        if(best.length<this.neighbors || d<best[best.length-1].distance){
          const hit={i,distance:d};
          let j=best.findIndex(x=>x.distance>d);
          if(j<0)j=best.length;
          best.splice(j,0,hit);
          if(best.length>this.neighbors)best.pop();
        }
      }
      const first=best[0].distance;
      let total=0;
      for(const h of best){
        const age=this.t-this.bank[h.i].t;
        h.weight=Math.exp(-(h.distance-first)/this.temperature)*Math.pow(0.5,age/this.halfLife);
        total+=h.weight;
      }
      for(const h of best){
        h.weight/=total||1;
        const value=this.bank[h.i].delta;
        for(let k=0;k<this.N;k++)result[k]+=value[k]*h.weight;
      }
      const confidence=Math.exp(-first/this.familiarityScale);
      return {delta:result,confidence, distance:first,
        hits:best.map(h=>({index:h.i,distance:h.distance,weight:h.weight,age:this.t-this.bank[h.i].t}))};
    }
    forecast(frame,address){
      const read=this.retrieve(address);
      const pred=new Float32Array(this.N),wake=new Float32Array(this.N);
      for(let i=0;i<this.N;i++){
        wake[i]=read.delta[i]*read.confidence;
        pred[i]=clamp(frame[i]+wake[i],0,1);
      }
      return {predicted:pred,wake,...read};
    }
    /** Surprise at t+1 edits past matched transitions and stores the completed bridge. */
    write(previousAddress,truthDelta,retrieval,surprise){
      // The newly arrived observation produced surprise = actual - prior forecast.
      // Only a sufficiently familiar past route receives this corrective excitation.
      if(this.feedback>0 && retrieval && retrieval.hits.length && retrieval.distance<0.01 && retrieval.confidence>0.4){
        const gain=this.feedback;
        for(const h of retrieval.hits){
          const existing=this.bank[h.index].delta;
          const amount=gain*h.weight;
          for(let k=0;k<this.N;k++)existing[k]+=amount*surprise[k];
        }
      }
      this.bank.push({key:previousAddress,delta:truthDelta,t:this.t});
      if(this.bank.length>this.capacity)this.bank.shift();
    }
    /** Push exactly one currently observed frame; no peeking at the future. */
    push(input){
      if(input.length!==this.N)throw Error('Expected '+this.N+' grayscale samples');
      const now=Float32Array.from(input);
      let completed=null;
      if(this.previous){
        const targetDelta=new Float32Array(this.N);
        for(let i=0;i<this.N;i++)targetDelta[i]=now[i]-this.previous[i];
        const error=new Float32Array(this.N);
        for(let i=0;i<this.N;i++)error[i]=now[i]-this.pending.predicted[i];
        const modelError=mse(this.pending.predicted,now),holdError=mse(this.previous,now);
        this.totalModel+=modelError;this.totalHold+=holdError;this.totalTrials++;
        completed={index:this.t,modelError,holdError,error,
          lastPrediction:this.pending.predicted,lastObserved:now,
          lastWake:this.pending.wake,lastConfidence:this.pending.confidence,
          improvement:holdError>1e-12?1-modelError/holdError:0};
        this.trials.push({t:this.t,modelError,holdError,confidence:this.pending.confidence,bankSize:this.bank.length});
        if(this.trials.length>5000)this.trials.shift();
        if(this.learning)this.write(this.previousAddress,targetDelta,this.pending,error);
      }
      const key=this.address(now,this.previous);
      const forecast=this.forecast(now,key);
      this.previous=now;
      this.previousAddress=key;
      this.pending=forecast;
      this.t++;
      this.latest=completed;
      return {completed,forecast,bankSize:this.bank.length,frames:this.t,
        meanModel:this.totalTrials?this.totalModel/this.totalTrials:null,
        meanHold:this.totalTrials?this.totalHold/this.totalTrials:null};
    }
    summary(){return {frames:this.t,completed:this.totalTrials,bankSize:this.bank.length,
      mode:this.mode,learning:this.learning,
      meanModel:this.totalTrials?this.totalModel/this.totalTrials:null,
      meanHold:this.totalTrials?this.totalHold/this.totalTrials:null,
      improvement:this.totalHold>0?1-this.totalModel/this.totalHold:null};}
  }
  ResidueHead.mse=mse;
  return ResidueHead;
});

