/* Original deterministic worlds, shared by offline browser and Node tests. */
(function(root,factory){const x=factory();if(typeof module!=='undefined'&&module.exports)module.exports=x;if(root)root.ResidueWorld=x;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const fract=x=>x-Math.floor(x);
  function render(kind,t,width=48,height=36){
    const frame=new Float32Array(width*height);
    let x,y,extra=false;
    if(kind==='bounce'){
      const p=fract(t/96),tri=1-Math.abs(2*p-1);
      x=7+(width-14)*tri;y=height*.5;
    }else if(kind==='orbit'){
      x=width*.5+width*.32*Math.cos(2*Math.PI*t/120);
      y=height*.5+height*.31*Math.sin(2*Math.PI*t/120);
      extra=true;
    }else if(kind==='split'){
      // Same location can be reached with opposite velocities: a hidden direction.
      const p=fract(t/80),tri=1-Math.abs(2*p-1);
      x=6+(width-12)*tri;y=height*.52;
      extra=true;
    }else throw Error('Unknown world '+kind);
    const s=3;
    for(let yy=0;yy<height;yy++)for(let xx=0;xx<width;xx++){
      const d2=(xx-x)**2+(yy-y)**2;
      let v=.08 + .86*Math.exp(-d2/(2*s*s));
      if(extra){const x2=width*.5+(width*.32)*Math.cos(2*Math.PI*(t+37)/120);
        const y2=height*.5+height*.27*Math.sin(2*Math.PI*(t+37)/120);
        v+=.25*Math.exp(-((xx-x2)**2+(yy-y2)**2)/(2*4*4));}
      frame[yy*width+xx]=Math.min(1,v);
    }
    return frame;
  }
  return {render};
});

