'use strict';
const assert=require('node:assert/strict');
const Head=require('./head.js');const World=require('./world.js');
function close(a,b,eps=1e-6){assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);}
function testCausality(){
 const h=new Head({width:12,height:8,gridX:3,gridY:2});
 const a=new Float32Array(96).fill(.1),b=new Float32Array(96).fill(.4);
 const one=h.push(a);assert.equal(one.completed,null);assert.equal(h.bank.length,0);
 close(Head.mse(one.forecast.predicted,a),0);
 const two=h.push(b);assert.equal(h.bank.length,1);
 close(h.bank[0].delta[0],.3);
 close(two.completed.modelError,two.completed.holdError);
}
function testDirectionalAddress(){
 const h=new Head({width:12,height:8,gridX:3,gridY:2,temperature:.0001,familiarityScale:1});
 const z=new Float32Array(96).fill(.5),plus=new Float32Array(96).fill(.6),minus=new Float32Array(96).fill(.4);
 const up=h.address(z,minus),down=h.address(z,plus);
 h.bank=[{key:up,delta:new Float32Array(96).fill(.1),t:0},{key:down,delta:new Float32Array(96).fill(-.1),t:0}];
 h.t=1;h.mode='temporal';
 assert.ok(h.retrieve(up).delta[0]>0.09);
 assert.ok(h.retrieve(down).delta[0]<-0.09);
 h.mode='gist';close(h.retrieve(up).delta[0],0,1e-6);
}
function testPredictionBeatsHold(){
 for(const kind of ['bounce','orbit']){
   const h=new Head({feedback:0}),per=[];
   for(let t=0;t<3*(kind==='orbit'?120:96);t++){
     const row=h.push(World.render(kind,t));
     if(row.completed&&t>=(kind==='orbit'?120:96)*2)per.push(row.completed);
   }
   const m=per.reduce((s,x)=>s+x.modelError,0)/per.length;
   const b=per.reduce((s,x)=>s+x.holdError,0)/per.length;
   console.log(kind,'model',m.toFixed(7),'hold',b.toFixed(7),'improvement',((1-m/b)*100).toFixed(2)+'%');
   assert.ok(m<b*.9,'Expected learned predictive advantage on repeating '+kind);
 }
}
function testUnseenNotOracular(){
 const h=new Head({feedback:0});
 for(let i=0;i<120;i++)h.push(World.render('orbit',i));
 const before=h.summary(),r=h.push(World.render('bounce',20));
 assert.equal(h.summary().completed,before.completed+1);
 assert.ok(r.completed.modelError>0);
}
testCausality();testDirectionalAddress();testPredictionBeatsHold();testUnseenNotOracular();
console.log('PASS: causal write, temporal address, periodic future prediction, unforeseen change');
