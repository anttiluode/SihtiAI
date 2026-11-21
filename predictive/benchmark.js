/* Independent one-cycle train / one-cycle frozen test. No lookahead. */
'use strict';
const fs=require('node:fs');
const Head=require('./head.js');
const World=require('./world.js');
const rows=[];
for(const [world,period] of [['bounce',96],['orbit',120]]){
  for(const address of ['temporal','gist']){
    const head=new Head({feedback:0});head.mode=address;
    const cases=[];
    for(let t=0;t<2*period;t++){
      if(t===period)head.learning=false;
      const r=head.push(World.render(world,t));
      if(t>=period+1&&r.completed)cases.push(r.completed);
    }
    const mse=cases.reduce((s,c)=>s+c.modelError,0)/cases.length;
    const hold=cases.reduce((s,c)=>s+c.holdError,0)/cases.length;
    const rec={world,address,trainFrames:period,testFrames:cases.length,
      learningFrozen:true,meanPredictionMSE:mse,meanHoldLastMSE:hold,
      relativeImprovement:1-mse/hold};rows.push(rec);
    console.log(`${world.padEnd(6)} ${address.padEnd(8)}  ${cases.length} frozen tests   head ${mse.toFixed(8)}  hold ${hold.toFixed(8)}  gain ${(100*(1-mse/hold)).toFixed(1)}%`);
  }
}
fs.mkdirSync('results',{recursive:true});
fs.writeFileSync('results/benchmark.json',JSON.stringify({protocol:'Train exactly one period; freeze; evaluate next cycle with no memory writes.',rows},null,2)+'\n');
